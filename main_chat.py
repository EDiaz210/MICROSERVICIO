# ============================================================================
# BACKEND CHAT - ESFOT - MAIN
# ============================================================================
# Sistema de chat con IA para responder preguntas de ESFOT
# - Búsqueda por similitud con spell correction y fuzzy matching
# - Sistema de calificaciones para respuestas
# - Módulo exclusivo para pasante (corrección de respuestas problemáticas)
# ============================================================================

# --- IMPORTS ---
import ast
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Endurecer  Chroma DB es  available locally (download from S3 if configured)
try:
    from download_chroma import ensure_chroma_local
    try:
        ensure_chroma_local()
    except Exception as _ex:
        print('[WARN] Could not ensure local Chroma DB:', _ex)
except Exception:
    # download_chroma may not be available or boto3 not installed; continue and let GestorEmbendings handle missing file
    print('[INFO] download_chroma not available - skipping S3 download step')


# Descarga la base de Chroma de S3 antes de iniciar el servicio
try:
    from download_chroma import ensure_chroma_local
    try:
        ensure_chroma_local()
    except Exception as _ex:
        print('[WARN] Could not ensure local Chroma DB:', _ex)
except Exception:
    print('[INFO] download_chroma not available - skipping S3 download step')

from gestor_embeddings import GestorEmbendings

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_words_from_qa():
    """Extrae palabras del dominio desde alimentar_datos_iniciales.py"""
    qa_words = set()
    qa_file = Path(__file__).parent / 'alimentar_datos_iniciales.py'
    if qa_file.exists():
        with open(qa_file, encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'preguntas_respuestas\s*=\s*(\[.*?\])\n\s*print', content, re.DOTALL)
        if match:
            try:
                preguntas_respuestas = ast.literal_eval(match.group(1))
                for qa in preguntas_respuestas:
                    for field in ['pregunta', 'respuesta', 'categoria']:
                        if field in qa:
                            tokens = re.findall(r'\b\w{3,}\b', qa[field].lower())
                            qa_words.update(tokens)
            except Exception as e:
                print(f"[WARN] No se pudo extraer palabras: {e}")
    return qa_words

def _normalizar_texto(texto):
    """Normaliza texto: quita tildes, puntuación, stopwords"""
    # Quitar tildes
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    # Quitar puntuación
    texto = re.sub(r'[\.,;:¿?¡!\-_/\\()\[\]{}"\'`]', '', texto)
    # Minúsculas
    texto = texto.lower()
    # Quitar stopwords
    stopwords = set(['el','la','los','las','de','del','a','en','y','o','u','que','por','para',
                    'con','sin','al','se','un','una','unos','unas','su','sus','mi','mis',
                    'tu','tus','es','son','como','cuál','cual','cuáles','cuales','qué','que',
                    'donde','dónde','cuando','cuándo','quien','quién','quienes','quiénes'])
    palabras = [p for p in texto.split() if p not in stopwords]
    texto = ' '.join(palabras)
    texto = texto.strip()
    return texto

# ============================================================================
# SPELL CHECKER & DOMAIN DICTIONARY
# ============================================================================

spell = SpellChecker(language='es')

# Blocklist de universidades - Se normalizan antes de comparar
UNIVERSIDADES_BLOQUEADAS = [
    "PUCE","POLI", "PUCESA", "Politécnica","UCE", "Salesiana", "San Francisco", "USFQ", 
    "ESPE", "ESPOCH", "ESPOL", "UTE", "UDLA", "UTPL", "UNEMI", "UNAE", "UNACH", "UNL", 
    "UNIVERSIDAD CATÓLICA", "UNIVERSIDAD CENTRAL", "UNIVERSIDAD DE GUAYAQUIL", 
    "UNIVERSIDAD DE CUENCA", "UNIVERSIDAD TÉCNICA", "UNIVERSIDAD NACIONAL", 
    "UNIVERSIDAD DEL AZUAY", "UNIVERSIDAD DEL PACÍFICO", "UNIVERSIDAD DE LAS AMÉRICAS", 
    "UNIVERSIDAD DE LOJA", "UNIVERSIDAD DE MANABÍ", "UNIVERSIDAD DE SANTA ELENA", 
    "UNIVERSIDAD DE SANTO DOMINGO"
]

# Normalizar universidades bloqueadas (sin tildes para comparación)
UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS = set()
for uni in UNIVERSIDADES_BLOQUEADAS:
    # Quitar tildes
    uni_norm = unicodedata.normalize('NFD', uni)
    uni_norm = ''.join(c for c in uni_norm if unicodedata.category(c) != 'Mn')
    UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS.add(uni_norm.upper())

# Construir diccionario de dominio
custom_words = set()
custom_words.update(extract_words_from_qa())
for uni in UNIVERSIDADES_BLOQUEADAS:
    custom_words.update(re.findall(r'\b\w{3,}\b', uni.lower()))

spell.word_frequency.load_words(custom_words)
domain_word_list = list(custom_words)

def corregir_ortografia(texto):
    """Corrige ortografía con spellchecker + fuzzy matching sobre dominio"""
    palabras = texto.split()
    corregidas = []
    for palabra in palabras:
        plower = palabra.lower()
        if plower not in spell:
            sugerida = spell.correction(palabra)
            if not sugerida or sugerida == palabra:
                # Fuzzy matching sobre palabras del dominio
                mejor, score, _ = process.extractOne(plower, domain_word_list, scorer=fuzz.ratio)
                if score >= 70:
                    corregidas.append(mejor)
                else:
                    corregidas.append(palabra)
            else:
                corregidas.append(sugerida)
        else:
            corregidas.append(palabra)
    return ' '.join(corregidas)

# ============================================================================
# FLASK APP & COMPONENTS
# ============================================================================

app = Flask(__name__)
CORS(app)

MODELO_QA = "distilbert-base-cased-distilled-squad"

# Inicializar componentes
gestor = GestorEmbendings()
qa_pipeline = pipeline(
    "question-answering",
    model=MODELO_QA,
    tokenizer=MODELO_QA
)

# ============================================================================
# CHAT BACKEND CLASS
# ============================================================================

class ChatBackend:
    """Backend principal para procesamiento de chat y preguntas"""
    
    def __init__(self):
        self.gestor = gestor
        self.qa_pipeline = qa_pipeline
    
    def eliminar_respuesta_pasante(self, id_respuesta):
        """Elimina una respuesta problemática por id"""
        try:
            return self.gestor.eliminar_respuesta_pasante(id_respuesta)
        except Exception as e:
            return {'success': False, 'error': f'Error al eliminar: {str(e)}'}
    
    def validar_pregunta_respuesta(self, pregunta, respuesta):
        """Valida campos con reglas flexibles"""
        errores = []
        if not pregunta or len(pregunta.strip()) == 0:
            errores.append("La pregunta es obligatoria")
        elif len(pregunta.strip()) < 2:
            errores.append("La pregunta debe tener al menos 2 caracteres")
        elif len(pregunta) > 300:
            errores.append("La pregunta no puede exceder 300 caracteres")

        if not respuesta or len(respuesta.strip()) == 0:
            errores.append("La respuesta es obligatoria")
        elif len(respuesta.strip()) < 2:
            errores.append("La respuesta debe tener al menos 2 caracteres")
        elif len(respuesta) > 500:
            errores.append("La respuesta no puede exceder 500 caracteres")

        return errores
    
    def agregar_pregunta_respuesta_pasante(self, pregunta, respuesta, categoria="General"):
        """El pasante agrega preguntas con calificación 5 por defecto"""
        errores = self.validar_pregunta_respuesta(pregunta, respuesta)
        if errores:
            return False, errores
        
        try:
            id_agregado = self.gestor.agregar_pregunta_respuesta(
                pregunta=pregunta.strip(),
                respuesta=respuesta.strip(),
                categoria=categoria,
                calificacion_inicial=5,
                es_pasante=True
            )
            return True, f"Pregunta y respuesta agregadas exitosamente (ID: {id_agregado})"
        except Exception as e:
            return False, [f"Error al guardar: {str(e)}"]
    
    def procesar_pregunta_chat(self, pregunta_usuario, usuario_id=None, rol_usuario="estudiante"):
        """Procesa preguntas del chat para estudiantes y administradores"""
        # Normalizar pregunta para verificar universidades bloqueadas
        pregunta_norm_check = unicodedata.normalize('NFD', str(pregunta_usuario).upper())
        pregunta_norm_check = ''.join(c for c in pregunta_norm_check if unicodedata.category(c) != 'Mn')
        
        # Verificar si menciona universidades bloqueadas
        for uni_bloqueada in UNIVERSIDADES_BLOQUEADAS_NORMALIZADAS:
            if uni_bloqueada in pregunta_norm_check:
                return {
                    'success': True,
                    'data': {
                        'respuesta': "No tengo información sobre esa universidad.",
                        'confianza': 'baja',
                        'acciones': ['contactar_administrador'],
                        'solicitar_calificacion': False
                    }
                }

        # Corregir ortografía
        pregunta_corregida = corregir_ortografia(str(pregunta_usuario))
        print(f"[DEBUG] Pregunta original: {pregunta_usuario}")
        print(f"[DEBUG] Pregunta corregida: {pregunta_corregida}")
        pregunta_norm = _normalizar_texto(pregunta_corregida)
        todas_preguntas = self.gestor.coleccion.get(where={"tipo": "pregunta_respuesta"}, include=["metadatas"])

        # Fuzzy matching
        preguntas_norm_bd = []
        for meta in todas_preguntas["metadatas"]:
            if meta:
                preguntas_norm_bd.append(_normalizar_texto(meta.get("pregunta", "")))

        FUZZY_UMBRAL = 70
        if preguntas_norm_bd:
            best_match, best_score, best_idx = process.extractOne(
                pregunta_norm, preguntas_norm_bd, scorer=fuzz.ratio
            ) if pregunta_norm else (None, 0, None)
            print(f"[FUZZY] Mejor coincidencia: '{best_match}' | Score: {best_score} | idx: {best_idx}")
            
            if best_score and best_score >= FUZZY_UMBRAL:
                meta = todas_preguntas["metadatas"][best_idx]
                confianza = 'alta' if best_score > 95 else 'media'
                respuesta_data = {
                    'respuesta': meta.get('respuesta', ''),
                    'confianza': confianza,
                    'id_respuesta': todas_preguntas['ids'][best_idx],
                    'score_final': 1.0,
                    'similitud': best_score / 100.0,
                    'calificacion_actual': meta.get('calificacion', 5),
                    'solicitar_calificacion': True,
                    'puede_reportar': True,
                    'opciones_reporte': [
                        "La información es incorrecta",
                        "No responde mi pregunta",
                        "Falta información importante",
                        "La información está desactualizada",
                        "Otro problema"
                    ]
                }
                print(f"[MATCH] Fuzzy aceptado")
                return {'success': True, 'data': respuesta_data}

        # Búsqueda por embedding
        EMBEDDING_UMBRAL = 0.1
        mejor_respuesta = self.gestor.buscar_mejor_respuesta(pregunta_corregida, umbral_confianza=0.0)
        if mejor_respuesta:
            print(f"[EMBEDDING] Similitud: {mejor_respuesta['similitud']:.2f}")

        if mejor_respuesta and mejor_respuesta['similitud'] >= EMBEDDING_UMBRAL:
            confianza = self._determinar_confianza(mejor_respuesta)
            respuesta_data = {
                'respuesta': mejor_respuesta['respuesta'],
                'confianza': confianza['nivel'],
                'id_respuesta': mejor_respuesta['id'],
                'score_final': mejor_respuesta['score_final'],
                'similitud': mejor_respuesta['similitud'],
                'calificacion_actual': mejor_respuesta['calificacion'],
                'solicitar_calificacion': True,
                'puede_reportar': True,
                'opciones_reporte': [
                    "La información es incorrecta",
                    "No responde mi pregunta",
                    "Falta información importante",
                    "La información está desactualizada",
                    "Otro problema"
                ]
            }
            return {'success': True, 'data': respuesta_data}

        # Usar IA con contexto
        contexto = ""
        if mejor_respuesta:
            resultados = self.gestor.coleccion.query(
                query_texts=[pregunta_corregida],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )
            contextos = []
            if resultados['documents'] and resultados['documents'][0]:
                for i, doc in enumerate(resultados['documents'][0]):
                    meta = resultados['metadatas'][0][i]
                    if meta.get('tipo') == 'pregunta_respuesta':
                        contextos.append(f"P: {meta.get('pregunta','')}. R: {meta.get('respuesta','')}")
            contexto = "\n".join(contextos)

        if contexto.strip():
            try:
                respuesta_ia = self.qa_pipeline({
                    'question': pregunta_corregida,
                    'context': contexto
                })
                respuesta_generada = respuesta_ia.get('answer', '').strip()
                if respuesta_generada and respuesta_generada.lower() not in ["no", "no se", "no sé", "no tengo información"] and len(respuesta_generada) > 3:
                    return {
                        'success': True,
                        'data': {
                            'respuesta': respuesta_generada,
                            'confianza': 'generada',
                            'acciones': [],
                            'solicitar_calificacion': True,
                            'puede_reportar': True
                        }
                    }
            except Exception as e:
                logger.error(f"Error generando respuesta IA: {e}")

        return {
            'success': True,
            'data': {
                'respuesta': "No tengo información para esa consulta.",
                'confianza': 'baja',
                'acciones': [],
                'solicitar_calificacion': False
            }
        }
    
    def _determinar_confianza(self, respuesta):
        """Determina nivel de confianza basado en similitud y calificación"""
        similitud = respuesta['similitud']
        calificacion = respuesta['calificacion']
        
        if similitud >= 0.9 and calificacion >= 4:
            return {'nivel': 'alta', 'motivo': 'Pregunta muy similar y buena calificación'}
        elif similitud >= 0.7 and calificacion >= 3:
            return {'nivel': 'media', 'motivo': 'Pregunta similar con calificación aceptable'}
        elif similitud >= 0.7 and calificacion < 3:
            return {'nivel': 'baja', 'motivo': 'Pregunta similar pero con baja calificación'}
        else:
            return {'nivel': 'baja', 'motivo': 'Baja similitud'}
    
    def calificar_respuesta(self, id_respuesta, calificacion, usuario_id=None, rol_usuario="estudiante"):
        """Califica respuesta (estudiantes y administradores)"""
        return self.gestor.calificar_respuesta(id_respuesta, calificacion, usuario_id, rol_usuario)
    
    def obtener_respuestas_problema(self):
        """Obtiene respuestas con calificación <3"""
        return self.gestor.obtener_respuestas_problema()
    
    def actualizar_respuesta_pasante(self, id_respuesta, nueva_respuesta, comentario_pasante=None):
        """El pasante actualiza una respuesta problemática"""
        return self.gestor.actualizar_respuesta_pasante(id_respuesta, nueva_respuesta, comentario_pasante)

# Inicializar backend
chat_backend = ChatBackend()

# ============================================================================
# ENDPOINTS - CHAT
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint principal para chat con soporte para streaming"""
    try:
        data = request.get_json()
        
        if not data or 'pregunta' not in data:
            return jsonify({
                'success': False,
                'error': 'Formato inválido',
                'message': 'Se requiere el campo "pregunta"'
            }), 400
        
        pregunta = data['pregunta'].strip()
        usuario_id = data.get('usuario_id')
        rol_usuario = data.get('rol', 'estudiante')
        streaming = data.get('streaming', False)
        
        if not pregunta:
            return jsonify({
                'success': False,
                'error': 'Pregunta vacía',
                'message': 'La pregunta no puede estar vacía'
            }), 400
        
        if streaming:
            def generate_stream():
                try:
                    yield f"data: {json.dumps({'etapa': 'buscando', 'mensaje': 'Buscando información relevante...'})}\n\n"
                    resultado = chat_backend.procesar_pregunta_chat(pregunta, usuario_id, rol_usuario)
                    yield f"data: {json.dumps({'etapa': 'procesando', 'mensaje': 'Generando respuesta...'})}\n\n"
                    
                    if resultado['success']:
                        data_res = resultado['data']
                        yield f"data: {json.dumps({
                            'etapa': 'completado',
                            'respuesta': data_res['respuesta'],
                            'confianza': data_res.get('confianza', 'media'),
                            'fuentes': [],
                            'id_respuesta_python': data_res.get('id_respuesta', ''),
                            'necesita_calificacion': data_res.get('solicitar_calificacion', True),
                            'puede_reportar': data_res.get('puede_reportar', True),
                            'calificacion_actual': data_res.get('calificacion_actual', 0)
                        })}\n\n"
                    else:
                        yield f"data: {json.dumps({'etapa': 'error', 'mensaje': 'Error procesando'})}\n\n"
                    
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error en streaming: {e}")
                    yield f"data: {json.dumps({'etapa': 'error', 'mensaje': 'Error interno'})}\n\n"
            
            return Response(generate_stream(), mimetype='text/plain')
        else:
            return jsonify(chat_backend.procesar_pregunta_chat(pregunta, usuario_id, rol_usuario))
        
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# ============================================================================
# ENDPOINTS - CALIFICACIÓN
# ============================================================================

@app.route('/api/calificar-respuesta', methods=['POST'])
def calificar_respuesta_endpoint():
    """Endpoint para calificar respuestas"""
    try:
        data = request.get_json()
        
        if not data or 'id_respuesta' not in data or 'calificacion' not in data:
            return jsonify({
                'success': False,
                'error': 'Se requiere id_respuesta y calificacion'
            }), 400
        
        resultado = chat_backend.calificar_respuesta(
            id_respuesta=data['id_respuesta'],
            calificacion=data['calificacion'],
            usuario_id=data.get('usuario_id'),
            rol_usuario=data.get('rol', 'estudiante')
        )
        
        enviado_a_correccion = False
        
        if resultado.get('success') and data['calificacion'] <= 3:
            pregunta_usuario = data.get('pregunta_usuario')
            respuesta_dada = data.get('respuesta_dada')
            
            if pregunta_usuario and respuesta_dada:
                print(f"CALIFICACIÓN BAJA ({data['calificacion']}) - Enviando al módulo de corrección")
                try:
                    resultado_correccion = gestor.enviar_a_modulo_correccion(
                        id_respuesta=data['id_respuesta'],
                        pregunta_usuario=pregunta_usuario,
                        respuesta_dada=respuesta_dada,
                        calificacion_recibida=data['calificacion']
                    )
                    if resultado_correccion.get('success'):
                        enviado_a_correccion = True
                except Exception as e:
                    print(f"Error al enviar al módulo de corrección: {e}")
        
        resultado['enviado_a_correccion'] = enviado_a_correccion
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

# ============================================================================
# ENDPOINTS - PASANTE
# ============================================================================

@app.route('/api/agregar-qa-pasante', methods=['POST'])
def agregar_qa_pasante_endpoint():
    """Endpoint EXCLUSIVO para pasante agregar preguntas"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Datos no proporcionados'}), 400
        
        pregunta = data.get('pregunta', '').strip()
        respuesta = data.get('respuesta', '').strip()
        categoria = data.get('categoria', 'General').strip()
        
        exito, resultado = chat_backend.agregar_pregunta_respuesta_pasante(pregunta, respuesta, categoria)
        
        if exito:
            return jsonify({
                'success': True,
                'message': resultado,
                'data': {
                    'pregunta': pregunta,
                    'categoria': categoria,
                    'longitud_respuesta': len(respuesta),
                    'calificacion_inicial': 5
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error de validación',
                'messages': resultado
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/pasante/respuestas-problema', methods=['GET'])
def respuestas_problema_endpoint():
    """Endpoint para pasante ver respuestas con calificación <3"""
    try:
        respuestas_problema = chat_backend.obtener_respuestas_problema()
        return jsonify({
            'success': True,
            'data': respuestas_problema,
            'total': len(respuestas_problema),
            'mensaje': f'Se encontraron {len(respuestas_problema)} respuestas problemáticas'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/pasante/actualizar-respuesta', methods=['POST'])
def actualizar_respuesta_endpoint():
    """Endpoint para pasante actualizar respuestas problemáticas"""
    try:
        data = request.get_json()
        if not data or 'id_respuesta' not in data or 'nueva_respuesta' not in data:
            return jsonify({
                'success': False,
                'error': 'Se requiere id_respuesta y nueva_respuesta'
            }), 400

        id_respuesta = data['id_respuesta']
        nueva_respuesta = data['nueva_respuesta'].strip()
        comentario_pasante = data.get('comentario_pasante')
        nueva_pregunta = data.get('pregunta_usuario')
        categoria = data.get('categoria', 'General')

        errores = []
        if not nueva_respuesta or len(nueva_respuesta) < 5:
            errores.append('La respuesta debe tener al menos 5 caracteres')
        elif len(nueva_respuesta) > 500:
            errores.append('La respuesta no puede exceder 500 caracteres')
        
        if errores:
            return jsonify({'success': False, 'error': 'Error de validación', 'messages': errores}), 400

        resultado_meta = gestor.coleccion.get(ids=[id_respuesta], include=["metadatas"])
        if not resultado_meta['metadatas'] or not resultado_meta['metadatas'][0]:
            return jsonify({'success': False, 'error': 'Respuesta no encontrada'}), 404
        
        metadata_actual = resultado_meta['metadatas'][0]
        pregunta_original = metadata_actual.get('pregunta', '').strip()

        if nueva_pregunta and nueva_pregunta.strip().lower() != pregunta_original.lower():
            resultado = gestor.agregar_nueva_pregunta_respuesta(
                pregunta_usuario=nueva_pregunta.strip(),
                respuesta_usuario=nueva_respuesta,
                categoria=categoria,
                es_pasante=True
            )
            if resultado.get('success'):
                return jsonify({
                    'success': True,
                    'message': 'Pregunta diferente, se creó un nuevo registro',
                    'accion': 'nueva',
                    'id_respuesta': resultado.get('id')
                })
            else:
                return jsonify({'success': False, 'error': resultado.get('error')}), 400
        else:
            resultado = chat_backend.actualizar_respuesta_pasante(
                id_respuesta=id_respuesta,
                nueva_respuesta=nueva_respuesta,
                comentario_pasante=comentario_pasante
            )
            return jsonify(resultado)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/pasante/eliminar-respuesta', methods=['POST', 'OPTIONS'])
def eliminar_respuesta_pasante_endpoint():
    """Endpoint para pasante eliminar respuestas problemáticas"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Obtener datos del request
        data = request.get_json(force=True, silent=True)
        print(f"[DEBUG] Request data: {data}")
        print(f"[DEBUG] Request content-type: {request.content_type}")
        
        if not data:
            print(f"[ERROR] No JSON data received")
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        if 'id_respuesta' not in data:
            print(f"[ERROR] Missing id_respuesta in data: {data}")
            return jsonify({'success': False, 'error': 'Se requiere id_respuesta'}), 400
        
        id_respuesta = data['id_respuesta']
        print(f"[INFO] Eliminando respuesta: {id_respuesta}")
        
        resultado = chat_backend.eliminar_respuesta_pasante(id_respuesta)
        print(f"[INFO] Resultado de eliminación: {resultado}")
        
        if resultado.get('success'):
            return jsonify({'success': True, 'message': resultado.get('mensaje', 'Eliminado')}), 200
        else:
            return jsonify({'success': False, 'error': resultado.get('error', 'Error al eliminar')}), 400
    except Exception as e:
        print(f"[ERROR] Exception in eliminar_respuesta_pasante_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pasante/procesar-respuesta-problema', methods=['POST'])
def procesar_respuesta_problema_endpoint():
    """Endpoint unificado para procesar respuestas problemáticas"""
    try:
        data = request.get_json()
        
        campos_requeridos = ['pregunta_usuario', 'nueva_respuesta', 'categoria']
        for campo in campos_requeridos:
            if campo not in data or not data[campo].strip():
                return jsonify({
                    'success': False,
                    'error': f'El campo {campo} es requerido'
                }), 400
        
        pregunta_usuario = data['pregunta_usuario'].strip()
        nueva_respuesta = data['nueva_respuesta'].strip()
        categoria = data['categoria'].strip()
        id_respuesta_existente = data.get('id_respuesta_existente')
        
        print(f" Procesando respuesta problema:")
        print(f"   Pregunta: {pregunta_usuario}")
        print(f"   ID existente: {id_respuesta_existente}")
        
        if id_respuesta_existente:
            print("🔄 Actualizando respuesta existente...")
            resultado = gestor.actualizar_respuesta_existente(
                id_respuesta=id_respuesta_existente,
                nueva_respuesta=nueva_respuesta,
                es_pasante=True
            )
        else:
            print(" Buscando pregunta exacta en BD...")
            busqueda_exacta = gestor.buscar_pregunta_exacta(pregunta_usuario)
            
            if busqueda_exacta['existe']:
                print(f"Pregunta existe, actualizando")
                resultado = gestor.actualizar_respuesta_existente(
                    id_respuesta=busqueda_exacta['id'],
                    nueva_respuesta=nueva_respuesta,
                    es_pasante=True
                )
            else:
                print(" Pregunta no existe, agregando nueva...")
                resultado = gestor.agregar_nueva_pregunta_respuesta(
                    pregunta_usuario=pregunta_usuario,
                    respuesta_usuario=nueva_respuesta,
                    categoria=categoria,
                    es_pasante=True
                )
        
        if resultado.get('success'):
            return jsonify({
                'success': True,
                'message': 'Respuesta procesada exitosamente',
                'accion': resultado.get('accion', 'procesada'),
                'id_respuesta': resultado.get('id')
            })
        else:
            return jsonify({
                'success': False,
                'error': resultado.get('error', 'Error al procesar')
            }), 400
            
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status_endpoint():
    """Endpoint de estado del sistema"""
    stats = gestor.obtener_estadisticas()
    
    return jsonify({
        'status': 'online',
        'modelo_qa': MODELO_QA,
        'base_conocimiento': {
            'total_documentos': stats['total_documentos'],
            'tipos': stats.get('por_tipo', {}),
            'categorias': stats.get('por_categoria', {})
        },
        'sistema_calificaciones': {
            'total_respuestas_calificadas': stats.get('total_respuestas_calificadas', 0),
            'calificacion_promedio': stats.get('calificacion_promedio', 0),
            'respuestas_problema': stats.get('respuestas_problema', 0)
        },
        'roles': {
            'estudiante': 'Usar chat, calificar (1-5), reportar problemas',
            'administrador': 'Usar chat, calificar (1-5), reportar problemas', 
            'pasante': 'Agregar preguntas (calificación 5), modificar respuestas <3'
        },
        'endpoints_pasante': {
            'agregar_pregunta': 'POST /api/agregar-qa-pasante',
            'ver_problemas': 'GET /api/pasante/respuestas-problema',
            'actualizar_respuesta': 'POST /api/pasante/actualizar-respuesta',
            'eliminar_respuesta': 'POST /api/pasante/eliminar-respuesta',
            'procesar_automatico': 'POST /api/pasante/procesar-respuesta-problema'
        }
    })

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print(" INICIANDO BACKEND CHAT - ESFOT")
    print("="*60)
    print(f" Modelo QA: {MODELO_QA}")
    
    stats = gestor.obtener_estadisticas()
    print(f" Base de conocimiento:")
    print(f"   - Documentos: {stats['total_documentos']}")
    print(f"   - Tipos: {stats.get('por_tipo', {})}")
    
    print(f" Sistema de Calificaciones:")
    print(f"   - Respuestas calificadas: {stats.get('total_respuestas_calificadas', 0)}")
    print(f"   - Calificación promedio: {stats.get('calificacion_promedio', 0)}")
    print(f"   - Respuestas problema (<3): {stats.get('respuestas_problema', 0)}")
    
    print(f" Endpoints disponibles:")
    print(f"   Chat:")
    print(f"   - POST /api/chat")
    print(f"   - POST /api/calificar-respuesta")
    print(f"   Pasante (respuestas problemáticas):")
    print(f"   - POST /api/agregar-qa-pasante")
    print(f"   - GET  /api/pasante/respuestas-problema")
    print(f"   - POST /api/pasante/actualizar-respuesta")
    print(f"   - POST /api/pasante/eliminar-respuesta")
    print(f"   - POST /api/pasante/procesar-respuesta-problema")
    print(f"   Sistema:")
    print(f"   - GET  /api/status")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
