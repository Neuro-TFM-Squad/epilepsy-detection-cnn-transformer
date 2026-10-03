# Índice de resultados y versiones

Conservar resultados por modelo/ejecución: configuraciones, referencias y
métricas permanecen juntas. No se han modificado números ni predicciones.

| Ubicación | Significado |
|---|---|
| `clinical_eval/{modelo}/final/`, `final_inv/` | fase inicial chb05/chb06 y dirección inversa |
| `clinical_eval/comparacion_final/` | resumen inicial; no incorpora EEGNet-8pat |
| `clinical_eval/eegnet_8pat/` | baseline posterior comparable, prueba chb06 |
| `clinical_eval/{attention_cnn,spatio_temporal_cnn,dwt}/nuevos_pacientes/` | ampliación chb11/chb12/chb13/chb15 |
| `clinical_eval/eegnet_8pat/nuevos_pacientes/` | EEGNet comparable en ampliación |
| `clinical_eval/{eegnet,dwt}/final_v2/` | posteriores; no sustituir automáticamente los principales |
| otras carpetas (`1`, `def*`, `robusto*`) | iteraciones/diagnósticos de postprocesado |
| `predictions/` | probabilidades locales ignoradas por Git |
| `xai/` | figuras IG públicas; atribuciones .npy conservadas solo en local |
| `benchmark/` | tiempos de inferencia y figura de coste |
| `logs/` | log histórico EEGNet-8pat conservado/versionable |

## Exportaciones frente a memoria final

Los JSON `nuevos_pacientes/metrics_global_test.json` usan 52 crisis (chb12: 17).
Una nota editorial conservada en local registra diez crisis adicionales en cuatro
registros chb12, sin nuevos TP ni cambios de FP/latencia. La memoria final
incorpora 62 crisis (chb12: 27); la corrección no llegó a estas exportaciones.

| Modelo | TP JSON | Sens. JSON (52) | Sens. memoria (62) | FA/h JSON |
|---|---:|---:|---:|---:|
| EEGNet-8pat | 2 | 3,85 % | 3,2 % | 0,4514 |
| AttentionCNN | 2 | 3,85 % | 3,2 % | 0,0000 |
| Híbrido | 3 | 5,77 % | 4,8 % | 0,1557 |
| DWT-MLP | 4 | 7,69 % | 6,5 % | 0,1323 |

Cifras de memoria: versión corregida documental, no una evaluación nueva.
Para reconciliar, generar exportación con referencias corregidas y preservar la
anterior. EEGNet-22pat incluye esos sujetos en entrenamiento y no es comparable.
El contexto de junio/borradores no representa el estado final. Ocho pacientes
comunes tampoco eliminan diferencias de EMA, calibración y selección de pesos.
