# TFM - Detección de Crisis Epilépticas Cross-Patient en EEG

## Contexto del proyecto
Este repositorio corresponde a un Trabajo de Fin de Máster sobre detección de crisis epilépticas cross-patient en señales EEG.

El objetivo principal no es alcanzar resultados SOTA, sino preservar un TFM terminado, sólido, defendible y metodológicamente consistente, priorizando:
- robustez cross-patient,
- utilidad clínica,
- sensibilidad por evento razonable,
- reducción de falsas alarmas por hora (FA/h),
- coherencia experimental,
- claridad en la redacción de la memoria.

## Datasets
- Dataset principal: CHB-MIT.
- Dataset auxiliar: Bonn.
- Bonn solo se usa como preentrenamiento / inicialización del bloque temporal en el modelo híbrido.
- No mezclar Bonn y CHB-MIT como si fueran el mismo dominio en el entrenamiento final.
- CHB-MIT es el único dominio clínico de evaluación principal.

## Modelos principales del TFM
1. EEGNet como baseline informativo.
2. Modelo híbrido espacio-temporal con transferencia desde Bonn.
3. Variante con atención ligera.
4. Enfoque frecuencial con DWT + clasificador.

## Reglas metodológicas clave
- La validación debe ser estrictamente cross-patient.
- La evaluación final es clínica, no solo por ventana.
- Métricas prioritarias:
  - sensibilidad por evento,
  - falsas alarmas por hora (FA/h),
  - latencia,
  - F1 clínico.
- El mejor modelo no debe elegirse por accuracy o loss de ventana, sino por un criterio clínico compuesto que penalice explícitamente las falsas alarmas.
- Antes de la evaluación clínica debe aplicarse postprocesado temporal.
- La calibración del score es parte legítima del pipeline si mejora el comportamiento clínico.
- La robustez metodológica y la utilidad clínica tienen prioridad sobre pequeñas mejoras numéricas aisladas.

## Entrenamiento
- Mantener entrenamiento conservador y estable.
- Evitar sobrecorregir el desbalance con varias técnicas agresivas simultáneamente.
- Si se usa `pos_weight`, debe limitarse.
- Gradient clipping es importante.
- Monitorizar estabilidad del score, NaNs y comportamiento en validación.
- Si hay transferencia, congelar y descongelar bloques de forma explícita y justificada.

## Estilo de ayuda esperado
Cuando trabajes en este repositorio:
- Primero analiza los archivos relevantes antes de proponer cambios.
- Explica primero qué hace el código y luego por qué importa metodológicamente.
- Antes de editar varios archivos o hacer cambios grandes, propón un plan breve.
- Mantén los cambios pequeños, localizados y fáciles de revisar.
- No reestructures el repositorio sin una razón clara.
- No propongas nuevos modelos o experimentos fuera del núcleo del TFM salvo petición explícita.
- Si una idea complica demasiado el cierre del TFM, indícalo claramente y propone una alternativa más simple.

## Ayuda con la memoria
Cuando ayudes a redactar la memoria:
- Usa estilo académico, técnico y claro, a nivel de máster en IA.
- Prioriza textos defendibles ante tribunal.
- No inventes detalles que no estén en el código, los resultados o los documentos.
- Si una sección está demasiado larga o redundante, señala qué recortar.
- Si una figura ayuda más que una explicación textual larga, indícalo.
- Mantén coherencia entre el código, los resultados y la memoria escrita.

## Navegación y preservación
- Leer primero `PROJECT_CONTEXT.md` y `CURRENT_STATE.md`.
- Consultar `ARCHITECTURE.md` para la etapa afectada; no escanear datos ni todo
  el repositorio sin necesidad. Mapa público en `docs/reorganization/moves.csv`;
  el inventario completo se conserva solo en local.
- Reutilizar módulos de `src/`; ejecutables en `scripts/` por etapa.
- Datos locales en `data/BONN` y `data/CHBMIT`; anotaciones en `data/metadata`.
- Pesos en `models/`, resultados en `experiments/`; memoria y material editorial
  en `docs/thesis/` solo en local. Respetar las exclusiones de publicación.
- No alterar resultados científicos, pesos ni anotaciones sin petición expresa.
  No sobrescribir ejecuciones históricas: usar destinos nuevos para reproducciones.
- No borrar artefactos sin comprobar su función ni decidir su vigencia por nombre.
- Mantener organización y documentación sincronizadas con cambios importantes.
- El TFM está terminado: no convertir notas antiguas en tareas pendientes actuales.
- Preservar particiones cross-patient y versiones de anotaciones. La memoria final
  corrige chb12 (62 crisis), mientras los JSON ampliados conservan 52.
- No importar preprocesadores, diagnósticos o scripts editoriales como prueba:
  algunos ejecutan trabajo al importar. Validar sintaxis sin ejecución.
- Los notebooks exploratorios no sustituyen el protocolo clínico principal.
- No introducir infraestructura o arquitectura futura sin necesidad.

## Qué NO hacer
- No perseguir indefinidamente mejoras marginales.
- No introducir arquitecturas muy complejas por inercia.
- No cambiar de dataset principal.
- No mezclar dominios de forma poco justificada.
- No ocultar resultados negativos: si un modelo no funciona bien, eso también es un resultado válido.