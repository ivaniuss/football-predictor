# Workflow — Football Predictor

## Flujo semanal

Solo necesitas ejecutar **2 notebooks**:

| Paso | Notebook | Cuándo |
|------|----------|--------|
| 1️⃣ | `06_update.ipynb` | Una vez por semana (o después de cada jornada) |
| 2️⃣ | `04_predict.ipynb` | Cada vez que quieras predecir un partido |

```
06_update.ipynb (5 celdas en orden)
│
├── Step 1: Descarga datos de football-data.co.uk
├── Step 2: Reconstruye features (form, ELO, H2H, referee)
├── Step 3: Evalúa modelo ACTUAL con partidos NUEVOS → tracking.csv
├── Step 4: Guarda features + Optuna tuning + Retrain → model.pkl
└── Step 5: Muestra historial de rendimiento acumulado

04_predict.ipynb
│
├── Carga model.pkl y features.csv
└── predict_match("Arsenal", "Man City", referee="M Oliver")
```

> ⚠️ Siempre ejecuta `06_update.ipynb` **antes** de predecir para que el modelo tenga los datos más recientes.

---

## Qué esperar en cada ejecución

### Primera vez

El Step 3 dirá:
```
ℹ️  First run — no previous model to evaluate against
```
Esto es normal — no hay modelo anterior contra el cual comparar. A partir de la segunda ejecución, el tracking empieza a funcionar.

### Ejecuciones siguientes

El Step 3 mostrará la evaluación retrospectiva:
```
📊 Retrospective evaluation on 10 NEW matches:
   Log-loss:  0.9821
   Accuracy:  0.5000 (5/10)
   Period:    2026-05-05 → 2026-05-11
   ✅ Saved to data/tracking.csv

       Date     HomeTeam       AwayTeam actual pred hit  P(H)   P(D)   P(A)
 2026-05-05      Arsenal       Man City      H    H   ✅  0.505  0.241  0.254
 2026-05-05    Liverpool        Chelsea      H    H   ✅  0.532  0.258  0.210
 2026-05-06      Everton       Man City      D    A   ❌  0.278  0.249  0.473
 ...
```

Y el Step 5 mostrará el historial acumulado:
```
📈 Model performance tracking (real-world, unseen matches):
eval_date   new_matches  log_loss  accuracy  period_start  period_end
2026-05-08           2    0.9821      0.50    2026-05-03  2026-05-04
2026-05-15          10    0.9654      0.55    2026-05-05  2026-05-11
2026-05-22          10    1.0102      0.40    2026-05-12  2026-05-18

   Cumulative avg log_loss:  0.9859
   Cumulative avg accuracy:  0.4833
   Total evaluated matches:  22
```

---

## Performance tracking (test set)

### ¿Qué es?

Cada semana, **antes de reentrenar**, el modelo actual predice los partidos nuevos que nunca vio durante entrenamiento ni tuning de hiperparámetros. Esos partidos actúan como un **test set natural y continuo**.

Los resultados se acumulan en `data/tracking.csv`.

### ¿Por qué es importante?

El log-loss de validación (el que reporta Optuna/TimeSeriesSplit durante el tuning) puede ser **optimista** porque los hiperparámetros se seleccionaron para minimizar exactamente esa métrica. El tracking mide el rendimiento **real** del modelo en datos que nunca influyeron en ninguna decisión.

### ¿Cómo interpretar los resultados?

| Señal | Significado |
|-------|------------|
| Tracking accuracy ≈ validation accuracy (~51%) | ✅ El modelo generaliza bien |
| Tracking accuracy mucho menor que validation | ⚠️ Hay overfitting de hiperparámetros |
| Tracking accuracy mejora con el tiempo | ✅ Más datos ayudan al modelo |
| Tracking accuracy empeora con el tiempo | ⚠️ Posible concept drift (el fútbol está cambiando) |

---

## Predicciones

```python
# Sin árbitro (usa promedios de liga)
predict_match("Arsenal", "Man City")

# Con árbitro
predict_match("Arsenal", "Man City", referee="M Oliver")
```

Output:
```
⚽ Arsenal vs Man City
─────────────────────────────
🏠 Arsenal wins:  50.5%
🤝 Draw:          24.1%
✈️  Man City wins: 25.4%
─────────────────────────────
📊 Prediction: Arsenal wins
```

---

## Los otros notebooks

| Notebook | Propósito | Cuándo usarlo |
|----------|-----------|---------------|
| `01_explore_data.ipynb` | Exploración de datos crudos | Solo para investigar |
| `02_feature_engineering.ipynb` | Diseño de features | Solo si agregas features nuevos |
| `03_model.ipynb` | Training inicial + evaluación + feature importance | Para experimentar desde cero |
| `05_tuning.ipynb` | Optuna con 50 trials (más profundo) | Solo si quieres un tuning exhaustivo |

> ⚠️ `03_model.ipynb` y `05_tuning.ipynb` sobreescriben `model.pkl`. El tracking seguirá funcionando, pero perderás los hiperparámetros del último Optuna semanal.

---

## Ciclo semana a semana

```
SEMANA 1:
  06_update → "First run, no previous model" → retrain → model_v1
  04_predict → predicciones con model_v1

SEMANA 2:
  06_update → evalúa model_v1 con partidos nuevos → tracking.csv
             → retrain → model_v2
  04_predict → predicciones con model_v2

SEMANA 3:
  06_update → evalúa model_v2 con partidos nuevos → tracking.csv (acumula)
             → retrain → model_v3
  04_predict → predicciones con model_v3
```

Cada versión del modelo se evalúa con datos que nunca vio durante entrenamiento ni durante tuning — ese es el principio del test set.
