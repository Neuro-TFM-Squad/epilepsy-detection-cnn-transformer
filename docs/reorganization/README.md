# Reorganización y validación

La reorganización separa módulos reutilizables (`src/`) de ejecutables por etapa
(`scripts/`), agrupa notebooks por finalidad y documenta el pipeline clínico real.
No cambia modelos, entrenamiento, pesos, anotaciones ni resultados científicos.

## Alcance público

Se publican código, dependencias, documentación técnica, notebooks históricos
identificados, métricas/configuraciones, checkpoints existentes y figuras.
La memoria, backups, revisiones, notas internas, scripts editoriales, inventario
local, exportaciones originales de máquina y atribuciones XAI completas se
conservan en local y quedan ignorados por Git.

El mapa de movimientos públicos está en [moves.csv](moves.csv). El inventario y
mapa completos permanecen en local. No se eliminó ningún archivo local.

## Compatibilidad y comprobaciones

Se actualizaron raíces `__file__`, imports Bonn y predictores legados, rutas de
anotaciones y benchmark, y la raíz de trabajo de notebooks válidos. Las antiguas
ubicaciones no tienen wrappers: actualizar automatizaciones con el mapa.
Las configuraciones y resultados históricos conservan sus rutas de procedencia.

[verification.json](verification.json) recoge la validación: sintaxis de 46
archivos Python, imports de 11 módulos seguros, ayudas CLI, carga y forward de
cinco checkpoints, prueba clínica sintética y conservación de outputs/metadatos
de seis notebooks. Se comprobó preservación de artefactos con hashes; los datasets
grandes se verificaron por presencia/tamaño. No se ejecutó entrenamiento,
preprocesado completo, notebooks ni instalación en un entorno limpio.

## Limitaciones conservadas

- Exportaciones clínicas ampliadas con 52 crisis frente a las 62 de la memoria
  final corregida: detalle en [índice de resultados](../../experiments/README.md).
- Notebook XAI legado con conflictos Git y notebooks exploratorios con splits
  por ventana; no sustituyen el protocolo clínico cross-patient.
- Índices históricos con rutas absolutas, filtro Bonn `.txt` frente a `.TXT` y
  variante de 2 s sin generador equivalente identificado.
- Dos evaluadores clínicos y variantes de entrenamiento no se consolidan sin
  comprobar equivalencia experimental. DWT actual escribe v2, mientras XAI y
  benchmark existentes usan el checkpoint anterior.

Continuación útil: generar una exportación clínica corregida en una ubicación
nueva y fijar un manifiesto de checkpoints, particiones y anotaciones.
