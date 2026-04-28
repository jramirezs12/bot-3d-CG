"""
train_intent.py — Entrena y evalúa el clasificador de intenciones.

Uso:
    python train_intent.py

Genera:
    data/intent_model.joblib   ← modelo entrenado listo para producción
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

from intent_classifier import (
    INTENT_DATA_PATH,
    MODEL_PATH,
    IntentClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_training_data(path: str) -> tuple[list[str], list[str]]:
    """Carga los datos de entrenamiento desde JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: list[str] = []
    labels: list[str] = []
    for intent, examples in data["intents"].items():
        for text in examples:
            texts.append(text)
            labels.append(intent)

    return texts, labels


def print_separator(char: str = "=", width: int = 65) -> None:
    print(char * width)


def print_section(title: str) -> None:
    print_separator()
    print(f"  {title}")
    print_separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n")
    print_section("ENTRENAMIENTO - CLASIFICADOR DE INTENCIONES E-COMMERCE (KTRONIX)")

    # ── Cargar datos ────────────────────────────────────────────────────────
    if not os.path.exists(INTENT_DATA_PATH):
        print(f"\nERROR: No se encontró el archivo de datos en:\n  {INTENT_DATA_PATH}")
        sys.exit(1)

    texts, labels = load_training_data(INTENT_DATA_PATH)

    print(f"\nDatos cargados: {len(texts)} ejemplos en total\n")

    counter = Counter(labels)
    print(f"{'Intención':<25} {'Ejemplos':>9}")
    print("-" * 37)
    for intent, count in sorted(counter.items()):
        bar = "#" * (count // 2)
        print(f"  {intent:<23} {count:>4}  {bar}")
    print()

    # ── Entrenar ────────────────────────────────────────────────────────────
    print("Entrenando clasificador TF-IDF + Regresión Logística...")
    clf = IntentClassifier()
    metrics = clf.train(texts, labels)

    # ── Reporte de clasificación ─────────────────────────────────────────────
    print("\n")
    print_section("REPORTE DE CLASIFICACIÓN (conjunto de prueba)")
    report = metrics["classification_report"]

    header = f"  {'Intención':<25} {'Precision':>9} {'Recall':>9} {'F1-Score':>9} {'Soporte':>9}"
    print(header)
    print("  " + "-" * 59)

    for intent in sorted(report.keys()):
        if intent in ("accuracy", "macro avg", "weighted avg"):
            continue
        v = report[intent]
        print(
            f"  {intent:<25} {v['precision']:>9.3f} {v['recall']:>9.3f}"
            f" {v['f1-score']:>9.3f} {int(v['support']):>9}"
        )

    print("  " + "-" * 59)
    w = report["weighted avg"]
    print(
        f"  {'PROMEDIO PONDERADO':<25} {w['precision']:>9.3f} {w['recall']:>9.3f}"
        f" {w['f1-score']:>9.3f}"
    )
    print(f"\n  Accuracy global: {report['accuracy']:.4f}  ({report['accuracy']*100:.2f}%)")
    print(f"  Train: {metrics['train_size']} muestras  |  Test: {metrics['test_size']} muestras")

    # ── Matriz de confusión ───────────────────────────────────────────────────
    print("\n")
    print_section("MATRIZ DE CONFUSIÓN")
    classes = metrics["classes"]
    cm = metrics["confusion_matrix"]

    col_w = 7
    print("  " + " " * 24 + "".join(f"{c[:col_w]:>{col_w}}" for c in classes))
    print("  " + "-" * (24 + col_w * len(classes)))
    for i, row in enumerate(cm):
        label = classes[i][:22]
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {label:<24}{row_str}")

    # ── Guardar modelo ────────────────────────────────────────────────────────
    clf.save()
    print(f"\n  Modelo guardado en: {MODEL_PATH}")

    # ── Pruebas rápidas ───────────────────────────────────────────────────────
    print("\n")
    print_section("PRUEBAS RÁPIDAS")

    test_queries = [
        ("Qué laptop me recomiendas para gaming?",         "recomendacion"),
        ("Cuánto cuesta el MacBook Pro?",                  "precio"),
        ("HP vs Dell, cuál es mejor?",                     "comparacion"),
        ("Mi laptop no enciende",                          "soporte_tecnico"),
        ("Tienen stock del Lenovo Legion?",                "disponibilidad"),
        ("Hola buenos días",                               "saludo"),
        ("Gracias, hasta luego",                           "despedida"),
        ("Cuáles son las specs del Dell XPS 15?",          "especificaciones"),
        ("Necesito algo para estudiar ingeniería",         "recomendacion"),
        ("Tienen garantía extendida?",                     "otro"),
    ]

    print(f"\n  {'Texto':<45} {'Predicho':<18} {'Real':<18} {'OK?'}")
    print("  " + "-" * 90)
    correct = 0
    for query, expected in test_queries:
        result = clf.predict(query)
        ok = "✓" if result["intent"] == expected else "✗"
        if result["intent"] == expected:
            correct += 1
        short_q = query[:43] + ".." if len(query) > 43 else query
        ok_str = "OK" if ok == "+" else "X"
        if result["intent"] == expected:
            ok_str = "OK"
        else:
            ok_str = "X "
        print(
            f"  {short_q:<45} {result['intent']:<18} {expected:<18} {ok_str}  "
            f"({result['confidence']:.2f})"
        )

    print(f"\n  Aciertos en pruebas: {correct}/{len(test_queries)}")
    print("\n¡Entrenamiento completado exitosamente!\n")


if __name__ == "__main__":
    main()
