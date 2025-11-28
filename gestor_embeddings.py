import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import json
from datetime import datetime
import uuid
import numpy as np
import re

# Definir las categorías permitidas
CATEGORIAS_PERMITIDAS = [
    "Quienes Somos", "Oferta Académica", "Admisiones", "Unidad Titulación", 
    "Estudiantes", "Comunidad", "Vinculacion Social", "Transparencia", 
    "Prácticas Pre-profesionales", "Calendario", "Contacto", "General", "Vinculación"
]

class GestorEmbendings:
    # Prior weight (pseudo-count) que representa cuánto pesa la calificación inicial del pasante.
    # Valores más altos hacen que la calificación inicial del pasante influya más y disminuya más lentamente.
    # Ajusta según política; aquí se incrementa para que el pasante sea más resistente a votos aislados.
    PRIOR_WEIGHT = 10.0

    def __init__(self, persist_directory="./chroma_data"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        self.coleccion_principal = "conocimiento_facultad"
        try:
            self.coleccion = self.client.get_collection(self.coleccion_principal)
        except:
            self.coleccion = self.client.create_collection(
                name=self.coleccion_principal,
                metadata={"description": "Base de conocimiento continua de la facultad"}
            )

    def _generar_id_unico(self, contenido, fuente):
        """Genera ID único basado en contenido y fuente"""
        contenido_hash = hashlib.md5(contenido.encode()).hexdigest()
        return f"{fuente}_{contenido_hash}"
    
    def _dividir_texto(self, texto, max_chars=1000):
        """Divide texto en chunks optimizados"""
        palabras = texto.split()
        chunks = []
        chunk_actual = []
        chars_actual = 0
        
        for palabra in palabras:
            if chars_actual + len(palabra) + 1 > max_chars:
                if chunk_actual:
                    chunks.append(" ".join(chunk_actual))
                chunk_actual = [palabra]
                chars_actual = len(palabra)
            else:
                chunk_actual.append(palabra)
                chars_actual += len(palabra) + 1
        
        if chunk_actual:
            chunks.append(" ".join(chunk_actual))
        
        return chunks
    
    def _calcular_similitud(self, texto1, texto2):
        """Calcula similitud entre textos"""
        emb1 = self.modelo_embedding.encode(texto1)
        emb2 = self.modelo_embedding.encode(texto2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def agregar_pregunta_respuesta(self, pregunta, respuesta, categoria="General", 
                                 calificacion_inicial=5, es_pasante=True):
        """Agrega preguntas y respuestas con sistema de calificación"""
        # VALIDAR CATEGORÍA
        if categoria not in CATEGORIAS_PERMITIDAS:
            raise ValueError(f"Categoría '{categoria}' no permitida. Las categorías válidas son: {', '.join(CATEGORIAS_PERMITIDAS)}")


        # Ya no se eliminan preguntas similares. Todas las preguntas se conservan.

        # AGREGAR NUEVA PREGUNTA-RESPUESTA CON CALIFICACIÓN
        contenido_completo = f"P: {pregunta}\nR: {respuesta}"
        id_unico = self._generar_id_unico(contenido_completo, "pregunta_respuesta")

        embedding = self.modelo_embedding.encode(contenido_completo).tolist()

        # Metadatos con sistema de calificación
        metadatos = {
            "tipo": "pregunta_respuesta",
            "categoria": categoria,
            "pregunta": pregunta,
            "respuesta": respuesta,
            "fuente": "pasante" if es_pasante else "usuario",
            "fecha_actualizacion": datetime.now().isoformat(),
            "version": "1.0",
            # `calificacion` refleja la calificación actual mostrada (inicialmente la del autor)
            "calificacion": calificacion_inicial,
            # No contar la calificación inicial del pasante como voto de usuario.
            # `total_calificaciones` y `suma_calificaciones` representan votos de usuarios reales.
            "total_calificaciones": 0,
            "suma_calificaciones": 0,
            # Prior count usado para suavizar la calificación inicial del pasante
            "prior_count": self.PRIOR_WEIGHT if es_pasante else 0,
            # Guardar la calificación que puso el pasante (si aplica) pero no como voto contado
            "calificacion_inicial_pasante": calificacion_inicial if es_pasante else None,
            "verificada": es_pasante,
            "es_pasante": es_pasante
        }

        self.coleccion.add(
            ids=[id_unico],
            documents=[contenido_completo],
            metadatas=[metadatos],
            embeddings=[embedding]
        )

        return id_unico
    
    def buscar_mejor_respuesta(self, consulta, umbral_confianza=0.75):
        """Busca la mejor respuesta considerando calificación y similitud"""
        try:
            resultados = self.coleccion.query(
                query_texts=[consulta],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )
            
            print(f"Búsqueda para: '{consulta}'")
            print(f"Resultados encontrados: {len(resultados['ids'][0]) if resultados['ids'] else 0}")
            
            mejores_respuestas = []
            
            if resultados['documents'] and resultados['documents'][0]:
                for i, distancia in enumerate(resultados['distances'][0]):
                    metadata = resultados['metadatas'][0][i]
                    
                    # Solo considerar preguntas/respuestas
                    if metadata.get('tipo') != 'pregunta_respuesta':
                        continue
                    
                    # Calcular score combinado
                    similitud = 1 - distancia
                    calificacion_normalizada = metadata.get('calificacion', 3) / 5
                    es_verificada = metadata.get('verificada', False)
                    es_pasante = metadata.get('es_pasante', False)
                    
                    # FÓRMULA INTELIGENTE
                    if similitud >= 0.8:
                        score_final = (similitud * 0.7) + (calificacion_normalizada * 0.3)
                    elif similitud >= 0.5:
                        score_final = (similitud * 0.5) + (calificacion_normalizada * 0.5)
                    else:
                        score_final = (similitud * 0.3) + (calificacion_normalizada * 0.7)
                    
                    # BONUS por respuestas verificadas y de pasante
                    if es_verificada:
                        score_final += 0.1
                    if es_pasante:
                        score_final += 0.15
                    
                    score_final = min(score_final, 1.0)
                    
                    print(f"Resultado {i}: '{metadata.get('pregunta', '')[:50]}...'")
                    print(f"Similitud: {similitud:.2f}, Calificación: {metadata.get('calificacion', 3)}")
                    print(f"Score final: {score_final:.2f}, Es pasante: {es_pasante}")
                    
                    if score_final >= umbral_confianza:
                        mejores_respuestas.append({
                            'id': resultados['ids'][0][i],
                            'pregunta': metadata.get('pregunta', ''),
                            'respuesta': metadata.get('respuesta', ''),
                            'score_final': score_final,
                            'similitud': similitud,
                            'calificacion': metadata.get('calificacion', 3),
                            'es_verificada': es_verificada,
                            'es_pasante': es_pasante,
                            'categoria': metadata.get('categoria', 'General'),
                            'total_calificaciones': metadata.get('total_calificaciones', 0)
                        })
            
            # Ordenar por score más alto
            if mejores_respuestas:
                mejores_respuestas.sort(key=lambda x: x['score_final'], reverse=True)
                mejor = mejores_respuestas[0]
                print(f" MEJOR RESPUESTA SELECCIONADA:")
                print(f"   Pregunta: '{mejor['pregunta']}'")
                print(f"   Score: {mejor['score_final']:.2f}")
                print(f"   Similitud: {mejor['similitud']:.2f}")
                print(f"   Calificación: {mejor['calificacion']}")
                print(f"   Es pasante: {mejor['es_pasante']}")
                return mejor
            
            print(" No se encontraron respuestas con umbral suficiente")
            return None
            
        except Exception as e:
            print(f"Error en búsqueda mejorada: {e}")
            return None

    def debug_ver_preguntas(self):
        """Función de debug para ver todas las preguntas en la base"""
        try:
            todos = self.coleccion.get(
                where={"tipo": "pregunta_respuesta"},
                include=["metadatas"]
            )
            
            print("="*50)
            print("🔍 DEBUG - TODAS LAS PREGUNTAS EN BASE:")
            print("="*50)
            
            for i, metadata in enumerate(todos['metadatas']):
                if metadata:
                    print(f"{i+1}. Pregunta: '{metadata.get('pregunta', '')}'")
                    print(f"   Respuesta: '{metadata.get('respuesta', '')}'")
                    print(f"   Categoría: {metadata.get('categoria', '')}")
                    print(f"   ID: {todos['ids'][i]}")
                    print(f"   Calificación: {metadata.get('calificacion', 'N/A')}")
                    print(f"   Es pasante: {metadata.get('es_pasante', False)}")
                    print("-" * 30)
            
            print(f"Total: {len(todos['ids'])} preguntas")
            print("="*50)
            
        except Exception as e:
            print(f"Error en debug: {e}")
    
    def calificar_respuesta(self, id_respuesta, calificacion, usuario_id=None, rol_usuario="estudiante"):
        """Califica una respuesta específica (1-5 estrellas) - Usuarios del chatbot"""
        
        if calificacion < 1 or calificacion > 5:
            return {'success': False, 'error': 'La calificación debe ser entre 1 y 5'}
        
        try:
            # Obtener metadata actual
            resultado = self.coleccion.get(ids=[id_respuesta], include=["metadatas"])

            if not resultado['metadatas'] or not resultado['metadatas'][0]:
                return {'success': False, 'error': 'Respuesta no encontrada'}

            metadata_actual = resultado['metadatas'][0]
            # DEBUG: imprimir metadatos actuales para facilitar diagnóstico
            try:
                print(f"DEBUG - Metadatos actuales para {id_respuesta}: {json.dumps(metadata_actual, ensure_ascii=False)}")
            except Exception:
                print(f"DEBUG - Metadatos actuales para {id_respuesta}: {metadata_actual}")

            # Robustecer tipos y valores por si faltan campos
            try:
                total_actual = int(metadata_actual.get('total_calificaciones', 0) or 0)
            except (ValueError, TypeError):
                total_actual = 0

            # suma_calificaciones puede no existir; intentar recomponerla desde promedio
            suma_val = metadata_actual.get('suma_calificaciones', None)
            if suma_val is None:
                # si existe calificacion y total, reconstruir suma
                try:
                    avg = float(metadata_actual.get('calificacion', 0) or 0)
                    suma_actual = round(avg * total_actual, 6)
                except (ValueError, TypeError):
                    suma_actual = 0.0
            else:
                try:
                    suma_actual = float(suma_val or 0)
                except (ValueError, TypeError):
                    suma_actual = 0.0

            # Evitar divisiones por cero: si total_actual negativo, resetear
            if total_actual < 0:
                total_actual = 0

            # Obtener prior (peso) y media prior (calificación inicial del pasante)
            try:
                prior_count = float(metadata_actual.get('prior_count', 0) or 0)
            except (ValueError, TypeError):
                prior_count = 0.0

            prior_mean = metadata_actual.get('calificacion_inicial_pasante') or metadata_actual.get('calificacion', 0)
            try:
                prior_mean = float(prior_mean)
            except (ValueError, TypeError):
                prior_mean = 0.0

            # Nuevo conteo/suma basado solo en votos de usuarios
            nuevo_total = total_actual + 1
            nueva_suma = suma_actual + float(calificacion)

            # Promedio suavizado con prior: (suma + prior_count*prior_mean) / (total + prior_count)
            denom = nuevo_total + prior_count
            if denom > 0:
                nueva_calificacion_promedio = round((nueva_suma + prior_count * prior_mean) / denom, 2)
            else:
                nueva_calificacion_promedio = round(prior_mean, 2)

            # Actualizar metadata (preservando campos relacionados con pasante)
            metadata_actualizada = {
                **metadata_actual,
                "calificacion": nueva_calificacion_promedio,
                "total_calificaciones": nuevo_total,
                "suma_calificaciones": nueva_suma,
                "prior_count": prior_count,
                "calificacion_inicial_pasante": metadata_actual.get('calificacion_inicial_pasante', metadata_actual.get('calificacion')),
                "ultima_calificacion": calificacion,
                "fecha_ultima_calificacion": datetime.now().isoformat(),
                "ultimo_calificador": rol_usuario
            }

            if usuario_id:
                metadata_actualizada["ultimo_usuario_calificador"] = usuario_id

            self.coleccion.update(
                ids=[id_respuesta],
                metadatas=[metadata_actualizada]
            )

            print(f" Calificación registrada: {calificacion} estrellas para respuesta {id_respuesta}")

            return {
                'success': True,
                'calificacion_promedio': nueva_calificacion_promedio,
                'total_calificaciones': nuevo_total,
                'mensaje': 'Calificación registrada exitosamente'
            }

        except Exception as e:
            return {'success': False, 'error': f'Error al calificar: {str(e)}'}
    
    # MÉTODOS EXCLUSIVOS PARA PASANTE - RESPUESTAS PROBLEMA
    def obtener_respuestas_problema(self):
        """Obtiene respuestas con calificación <3 o marcadas para corrección"""
        try:
            todas_respuestas = self.coleccion.get(
                where={"tipo": "pregunta_respuesta"},
                include=["metadatas"]
            )
            
            respuestas_problema = []
            for i, metadata in enumerate(todas_respuestas['metadatas']):
                #  INCLUIR: calificación < 3 O marcadas para corrección (mostrar solo estrictamente menores a 3)
                if metadata and (
                    (isinstance(metadata.get('calificacion', None), (int, float)) and metadata.get('calificacion') < 3) or
                    metadata.get('requiere_correccion', False)
                ):
                    respuestas_problema.append({
                        'id': todas_respuestas['ids'][i],
                        'pregunta': metadata.get('pregunta', ''),
                        'respuesta_actual': metadata.get('respuesta', ''),
                        'calificacion_actual': metadata.get('calificacion', 0),
                        'total_calificaciones': metadata.get('total_calificaciones', 0),
                        'categoria': metadata.get('categoria', 'General'),
                        'fecha_ultima_actualizacion': metadata.get('fecha_actualizacion', ''),
                        'fuente': metadata.get('fuente', 'desconocida'),
                        'es_pasante': metadata.get('es_pasante', False),
                        'requiere_correccion': metadata.get('requiere_correccion', False),
                        'pregunta_usuario_original': metadata.get('pregunta_usuario_original', ''),
                        'calificacion_que_activo_correccion': metadata.get('calificacion_que_activo_correccion', 0)
                    })
            
            print(f" Respuestas problema encontradas: {len(respuestas_problema)}")
            return respuestas_problema
            
        except Exception as e:
            print(f"Error al obtener respuestas problema: {e}")
            return []

    def reparar_calificaciones_pasantes(self, dry_run=True):
        """
        Repara entradas creadas por pasantes donde la calificación inicial fue contada
        como un voto (total_calificaciones==1 y suma_calificaciones==calificacion).
        Si dry_run=True solo muestra cuántos se repararían.
        """
        try:
            todos = self.coleccion.get(where={"tipo": "pregunta_respuesta"}, include=["metadatas", "ids"])
            reparados = 0
            afectados = []
            for idx, meta in enumerate(todos.get('metadatas', [])):
                if not meta:
                    continue
                es_pasante = meta.get('es_pasante', False)
                total = int(meta.get('total_calificaciones', 0) or 0)
                suma = meta.get('suma_calificaciones', None)
                cal = meta.get('calificacion', None)

                # condición típica: creado por pasante y tiene total==1 y suma equals cal
                if es_pasante and total == 1 and suma is not None:
                    try:
                        suma_f = float(suma)
                        cal_f = float(cal) if cal is not None else None
                    except (ValueError, TypeError):
                        continue

                    if cal_f is not None and abs(suma_f - (cal_f * total)) < 1e-6:
                        affected_id = todos['ids'][idx]
                        reparados += 1
                        afectados.append(affected_id)
                        if not dry_run:
                            nueva_meta = dict(meta)
                            nueva_meta['total_calificaciones'] = 0
                            nueva_meta['suma_calificaciones'] = 0
                            # preservar la calificación original del pasante
                            nueva_meta['calificacion_inicial_pasante'] = cal_f
                            # establecer prior_count si no existía (suavizado)
                            if 'prior_count' not in nueva_meta or nueva_meta.get('prior_count') is None:
                                nueva_meta['prior_count'] = float(self.PRIOR_WEIGHT if meta.get('es_pasante', False) else 0)
                            else:
                                try:
                                    nueva_meta['prior_count'] = float(nueva_meta.get('prior_count') or 0)
                                except (ValueError, TypeError):
                                    nueva_meta['prior_count'] = float(self.PRIOR_WEIGHT if meta.get('es_pasante', False) else 0)
                            # mantener `calificacion` para mostrar, pero usuarios reales contarán desde 0
                            self.coleccion.update(ids=[affected_id], metadatas=[nueva_meta])

            print(f"Reparación pasantes: encontrados={len(todos.get('metadatas', []))}, a_reparar={reparados}, dry_run={dry_run}")
            if reparados > 0:
                print("IDs afectados:")
                for a in afectados:
                    print(" - ", a)
            return {'found': len(todos.get('metadatas', [])), 'to_repair': reparados, 'ids': afectados}
        except Exception as e:
            print(f"Error en reparar_calificaciones_pasantes: {e}")
            return {'error': str(e)}
    
    def buscar_pregunta_exacta(self, pregunta_usuario):
        """Busca si existe exactamente la misma pregunta del usuario"""
        try:
            todas_preguntas = self.coleccion.get(
                where={"tipo": "pregunta_respuesta"},
                include=["metadatas", "documents"]
            )
            
            for i, metadata in enumerate(todas_preguntas['metadatas']):
                if metadata and metadata.get('pregunta', '').strip().lower() == pregunta_usuario.strip().lower():
                    return {
                        'id': todas_preguntas['ids'][i],
                        'pregunta_bd': metadata.get('pregunta', ''),
                        'respuesta_bd': metadata.get('respuesta', ''),
                        'existe': True
                    }
            return {'existe': False}
        except Exception as e:
            print(f"Error en buscar_pregunta_exacta: {e}")
            return {'existe': False}

    def agregar_nueva_pregunta_respuesta(self, pregunta_usuario, respuesta_usuario, categoria="General", es_pasante=False):
        """Agrega una nueva pregunta-respuesta cuando no existe"""
        try:
            # Validar categoría
            if categoria not in CATEGORIAS_PERMITIDAS:
                categoria = "General"
            
            # Calificación inicial: 5 para pasante
            calificacion_inicial = 5
            
            id_agregado = self.agregar_pregunta_respuesta(
                pregunta=pregunta_usuario.strip(),
                respuesta=respuesta_usuario.strip(),
                categoria=categoria,
                calificacion_inicial=calificacion_inicial,
                es_pasante=es_pasante
            )
            
            return {
                'success': True, 
                'id': id_agregado,
                'accion': 'agregada'
            }
            
        except Exception as e:
            print(f" Error en agregar_nueva_pregunta_respuesta: {e}")
            return {'success': False, 'error': str(e)}

    def actualizar_respuesta_existente(self, id_respuesta, nueva_respuesta, es_pasante=False):
        """Actualiza una respuesta existente - CORREGIDO"""
        try:
            # Obtener metadata actual
            resultado = self.coleccion.get(ids=[id_respuesta], include=["metadatas"])
            if not resultado['metadatas']:
                return {'success': False, 'error': 'Respuesta no encontrada'}
            
            metadata_actual = resultado['metadatas'][0]
            pregunta_original = metadata_actual.get('pregunta', '')
            
            # CORREGIR: Manejo robusto de versión
            version_actual = metadata_actual.get('version', '1')
            try:
                if isinstance(version_actual, str):
                    version_num = int(float(version_actual))
                else:
                    version_num = int(version_actual)
                nueva_version = str(version_num + 1)
            except (ValueError, TypeError):
                nueva_version = "2"  # Valor por defecto
            
            # Actualizar metadata
            metadata_actualizada = {
                **metadata_actual,
                "respuesta": nueva_respuesta,
                "fecha_actualizacion": datetime.now().isoformat(),
                "version": nueva_version,
                "mejorada_por_pasante": es_pasante
            }
            
            # Si es pasante, resetear calificación
            if es_pasante:
                metadata_actualizada.update({
                    "calificacion": 5,
                    # No contar la calificación del pasante como voto de usuario
                    "total_calificaciones": 0,
                    "suma_calificaciones": 0,
                    "calificacion_inicial_pasante": 5,
                    "prior_count": float(self.PRIOR_WEIGHT),
                    "verificada": True
                })
            
            # Actualizar documento
            nuevo_contenido = f"P: {pregunta_original}\nR: {nueva_respuesta}"
            
            self.coleccion.update(
                ids=[id_respuesta],
                documents=[nuevo_contenido],
                metadatas=[metadata_actualizada]
            )
            
            # Actualizar embedding
            nuevo_embedding = self.modelo_embedding.encode(nuevo_contenido).tolist()
            self.coleccion.update(
                ids=[id_respuesta],
                embeddings=[nuevo_embedding]
            )
            
            print(f" Respuesta actualizada: {id_respuesta}")
            
            return {
                'success': True,
                'mensaje': 'Respuesta actualizada exitosamente',
                'id_respuesta': id_respuesta,
                'accion': 'actualizada'
            }
            
        except Exception as e:
            print(f"Error en actualizar_respuesta_existente: {e}")
            return {'success': False, 'error': f'Error al actualizar: {str(e)}'}

    def enviar_a_modulo_correccion(self, id_respuesta, pregunta_usuario, respuesta_dada, calificacion_recibida):
        """
        Envía automáticamente una pregunta/respuesta al módulo de corrección
        cuando recibe una calificación ≤ 3
        """
        try:
            print(f" MÓDULO DE CORRECCIÓN AUTOMÁTICA ACTIVADO")
            print(f"   ID Respuesta: {id_respuesta}")
            print(f"   Pregunta: {pregunta_usuario}")
            print(f"   Calificación: {calificacion_recibida}")
            
            # Obtener metadata actual
            resultado = self.coleccion.get(ids=[id_respuesta], include=["metadatas"])
            if not resultado['metadatas'] or not resultado['metadatas'][0]:
                print(f"No se encontró la respuesta en la BD")
                return {'success': False, 'error': 'Respuesta no encontrada'}
            
            metadata_actual = resultado['metadatas'][0]
            
            # Marcar como "requiere corrección"
            metadata_actualizada = {
                **metadata_actual,
                "requiere_correccion": True,
                "fecha_marcada_correccion": datetime.now().isoformat(),
                "calificacion_que_activo_correccion": calificacion_recibida,
                "pregunta_usuario_original": pregunta_usuario,
                "respuesta_dada_original": respuesta_dada
            }
            
            self.coleccion.update(
                ids=[id_respuesta],
                metadatas=[metadata_actualizada]
            )
            
            print(f"Respuesta marcada para corrección en ChromaDB")
            print(f"   El pasante podrá verla en el módulo de corrección")
            
            return {
                'success': True,
                'mensaje': 'Respuesta enviada al módulo de corrección',
                'id_respuesta': id_respuesta,
                'requiere_atencion_pasante': True
            }
            
        except Exception as e:
            print(f"Error en enviar_a_modulo_correccion: {e}")
            return {'success': False, 'error': f'Error al enviar a corrección: {str(e)}'}

    def actualizar_respuesta_pasante(self, id_respuesta, nueva_respuesta):
        """El pasante actualiza una respuesta problemática"""
        try:
            # Obtener metadata actual
            resultado = self.coleccion.get(ids=[id_respuesta], include=["metadatas"])
            if not resultado['metadatas']:
                return {'success': False, 'error': 'Respuesta no encontrada'}
            
            metadata_actual = resultado['metadatas'][0]
            
            # Actualizar respuesta y resetear calificación
            # Manejar versión robustamente
            version_actual = metadata_actual.get('version', '1')
            try:
                version_num = int(float(version_actual))
                nueva_version = str(version_num + 1)
            except (ValueError, TypeError):
                nueva_version = "2"
            metadata_actualizada = {
                **metadata_actual,
                "respuesta": nueva_respuesta,
                "calificacion": 5,  # Resetear a 5 después de la mejora
                # No contar la calificación del pasante como voto de usuario
                "total_calificaciones": 0,
                "suma_calificaciones": 0,
                "calificacion_inicial_pasante": 5,
                "prior_count": float(self.PRIOR_WEIGHT),
                "fecha_actualizacion": datetime.now().isoformat(),
                "version": nueva_version,
                "mejorada_por_pasante": True,
                "requiere_correccion": False  #  Quitar marca de corrección
            }
            
            # Actualizar el documento también
            nuevo_contenido = f"P: {metadata_actual['pregunta']}\nR: {nueva_respuesta}"
            
            self.coleccion.update(
                ids=[id_respuesta],
                documents=[nuevo_contenido],
                metadatas=[metadata_actualizada]
            )
            
            # Actualizar embedding
            nuevo_embedding = self.modelo_embedding.encode(nuevo_contenido).tolist()
            self.coleccion.update(
                ids=[id_respuesta],
                embeddings=[nuevo_embedding]
            )
            
            print(f"Pasante actualizó respuesta: {id_respuesta}")
            
            return {
                'success': True,
                'mensaje': 'Respuesta actualizada exitosamente.',
                'id_respuesta': id_respuesta
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Error al actualizar: {str(e)}'}

    def eliminar_respuesta_pasante(self, id_respuesta):
        """Elimina una respuesta problemática por id en ChromaDB"""
        try:
            self.coleccion.delete(ids=[id_respuesta])
            print(f"Respuesta eliminada: {id_respuesta}")
            return {'success': True, 'mensaje': 'Respuesta eliminada correctamente', 'id_respuesta': id_respuesta}
        except Exception as e:
            print(f"Error al eliminar respuesta: {e}")
            return {'success': False, 'error': f'Error al eliminar: {str(e)}'}
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas del conocimiento incluyendo calificaciones"""
        total_docs = self.coleccion.count()
        
        if total_docs == 0:
            return {"total_documentos": 0}
        
        todos_datos = self.coleccion.get(include=["metadatas"])
        metadatas = todos_datos["metadatas"]
        
        estadisticas = {
            "total_documentos": total_docs,
            "por_tipo": {},
            "por_categoria": {},
            "por_fuente": {},
            "total_respuestas_calificadas": 0,
            "calificacion_promedio": 0,
            "respuestas_problema": 0
        }
        
        calificaciones_totales = 0
        respuestas_calificadas = 0
        respuestas_problema = 0
        
        for meta in metadatas:
            tipo = meta.get('tipo', 'desconocido')
            estadisticas["por_tipo"][tipo] = estadisticas["por_tipo"].get(tipo, 0) + 1
            
            categoria = meta.get('categoria', 'general')
            estadisticas["por_categoria"][categoria] = estadisticas["por_categoria"].get(categoria, 0) + 1
            
            fuente = meta.get('fuente', 'desconocida')
            estadisticas["por_fuente"][fuente] = estadisticas["por_fuente"].get(fuente, 0) + 1
            
            # Estadísticas de calificaciones
            if meta.get('total_calificaciones', 0) > 0:
                respuestas_calificadas += 1
                calificaciones_totales += meta.get('calificacion', 0)
                
                if meta.get('calificacion', 5) < 3:
                    respuestas_problema += 1
        
        if respuestas_calificadas > 0:
            estadisticas["total_respuestas_calificadas"] = respuestas_calificadas
            estadisticas["calificacion_promedio"] = round(calificaciones_totales / respuestas_calificadas, 2)
        
        estadisticas["respuestas_problema"] = respuestas_problema
        
        return estadisticas