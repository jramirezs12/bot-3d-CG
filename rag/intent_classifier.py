"""
intent_classifier.py — Clasificador de intenciones para e-commerce de laptops (Ktronix).

Arquitectura:
  TF-IDF (bigramas) + Regresión Logística (sklearn)

Intenciones:
  recomendacion   — el usuario quiere que le recomienden un producto
  especificaciones — el usuario pregunta por specs de un producto
  precio          — el usuario pregunta por precios o financiación
  comparacion     — el usuario quiere comparar productos
  disponibilidad  — el usuario pregunta por stock o entrega
  soporte_tecnico — el usuario tiene un problema técnico
  saludo          — saludo inicial
  despedida       — cierre de conversación
  otro            — consultas generales no clasificadas
"""

from __future__ import annotations

import json
import os
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "intent_model.joblib")
INTENT_DATA_PATH = os.path.join(DATA_DIR, "intent_data.json")

# ---------------------------------------------------------------------------
# Etiquetas de intención
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "recomendacion",
    "especificaciones",
    "precio",
    "comparacion",
    "disponibilidad",
    "soporte_tecnico",
    "saludo",
    "despedida",
    "otro",
]

# Intenciones que requieren consulta al RAG de productos
INTENTS_NEED_RAG: set[str] = {
    "recomendacion",
    "especificaciones",
    "precio",
    "comparacion",
    "disponibilidad",
}

# Prompts de sistema específicos por intención
INTENT_SYSTEM_PROMPTS: dict[str, str] = {
    "recomendacion": (
        "Eres un asesor experto de Ktronix, tienda de tecnología colombiana. "
        "El usuario quiere una recomendación de laptop. Usa la información del catálogo "
        "recuperada para dar una recomendación personalizada y concreta. Menciona "
        "el nombre del modelo, precio en pesos colombianos y por qué es la mejor opción "
        "para las necesidades del usuario."
    ),
    "especificaciones": (
        "Eres un asesor técnico de Ktronix. El usuario pregunta por especificaciones "
        "de un producto. Usa la información del catálogo para dar los detalles técnicos "
        "precisos: procesador, RAM, almacenamiento, pantalla, batería y puertos."
    ),
    "precio": (
        "Eres un asesor comercial de Ktronix. El usuario pregunta por precios. "
        "Proporciona los precios en pesos colombianos (COP) con base en el catálogo. "
        "Menciona opciones de financiación si aplica."
    ),
    "comparacion": (
        "Eres un asesor experto de Ktronix. El usuario quiere comparar productos. "
        "Usa la información del catálogo para hacer una comparación clara y objetiva. "
        "Presenta pros y contras de cada opción y da una recomendación final."
    ),
    "disponibilidad": (
        "Eres un asesor de ventas de Ktronix. El usuario pregunta por disponibilidad "
        "o tiempos de entrega. Usa el catálogo para informar el stock actual y "
        "ofrecer alternativas si el producto no está disponible."
    ),
    "soporte_tecnico": (
        "Eres un técnico especializado de Ktronix. El usuario tiene un problema "
        "con su laptop. Da instrucciones claras paso a paso para resolver el problema. "
        "Si el problema es grave, recomienda llevar el equipo al servicio técnico."
    ),
    "saludo": (
        "Eres el asistente virtual de Ktronix, tienda de tecnología colombiana. "
        "Saluda al usuario de forma amable y pregúntale en qué puedes ayudarle. "
        "Menciona que puedes ayudar con recomendaciones, precios, especificaciones "
        "y soporte técnico de laptops."
    ),
    "despedida": (
        "Eres el asistente de Ktronix. El usuario se está despidiendo. "
        "Despídete de forma cálida, agradece su visita y recuérdale que puede "
        "volver cuando necesite más ayuda."
    ),
    "otro": (
        "Eres el asistente virtual de Ktronix, tienda de tecnología colombiana. "
        "Responde de forma amable y útil. Si la consulta está fuera de tu dominio, "
        "ofrece orientar al usuario con la tienda."
    ),
}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Clasificador de intenciones basado en TF-IDF + Regresión Logística."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.classes: list[str] = []

    # ---- Preprocesamiento ------------------------------------------------

    @staticmethod
    def _preprocess(text: str) -> str:
        """Normaliza el texto: minúsculas, elimina puntuación extra."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    # ---- Entrenamiento ---------------------------------------------------

    def train(self, texts: list[str], labels: list[str]) -> dict:
        """
        Entrena el clasificador y retorna un diccionario con métricas.

        Returns:
            {
              classification_report: dict,
              confusion_matrix: list[list[int]],
              classes: list[str],
              train_size: int,
              test_size: int,
              accuracy: float,
            }
        """
        texts_clean = [self._preprocess(t) for t in texts]

        X_train, X_test, y_train, y_test = train_test_split(
            texts_clean,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=8000,
                        min_df=1,
                        sublinear_tf=True,
                        analyzer="word",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=5.0,
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

        self.pipeline.fit(X_train, y_train)
        self.classes = list(self.pipeline.classes_)

        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=self.classes).tolist()

        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "classes": self.classes,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": report["accuracy"],
        }

    # ---- Predicción ------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Clasifica la intención de un texto.

        Returns:
            {
              intent: str,
              confidence: float,
              scores: dict[str, float],
              needs_rag: bool,
              system_prompt: str,
            }
        """
        if self.pipeline is None:
            raise RuntimeError(
                "El clasificador no está entrenado. Ejecuta train_intent.py primero."
            )

        text_clean = self._preprocess(text)
        intent: str = self.pipeline.predict([text_clean])[0]
        proba: np.ndarray = self.pipeline.predict_proba([text_clean])[0]

        scores = {cls: float(p) for cls, p in zip(self.classes, proba)}
        confidence = float(proba.max())

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "scores": scores,
            "needs_rag": intent in INTENTS_NEED_RAG,
            "system_prompt": INTENT_SYSTEM_PROMPTS.get(intent, INTENT_SYSTEM_PROMPTS["otro"]),
        }

    # ---- Persistencia ----------------------------------------------------

    def save(self, path: str = MODEL_PATH) -> None:
        """Guarda el modelo entrenado en disco."""
        if self.pipeline is None:
            raise RuntimeError("No hay modelo entrenado para guardar.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes}, path)
        print(f"[OK] Modelo guardado en {path}")

    def load(self, path: str = MODEL_PATH) -> bool:
        """
        Carga el modelo desde disco.

        Returns:
            True si se cargó correctamente, False si no existe el archivo.
        """
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.classes = data["classes"]
        return True


# ---------------------------------------------------------------------------
# Singleton para uso en FastAPI
# ---------------------------------------------------------------------------

_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    """Retorna la instancia singleton del clasificador (carga el modelo si existe)."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
        if not _classifier.load():
            print(
                "ADVERTENCIA: No se encontró modelo de intenciones. "
                "Ejecuta train_intent.py primero."
            )
    return _classifier


def classify_intent(text: str) -> dict:
    """
    Función de conveniencia para clasificar la intención de un texto.
    Retorna 'otro' con confianza 0 si el modelo no está disponible.
    """
    clf = get_classifier()
    try:
        return clf.predict(text)
    except RuntimeError:
        return {
            "intent": "otro",
            "confidence": 0.0,
            "scores": {},
            "needs_rag": False,
            "system_prompt": INTENT_SYSTEM_PROMPTS["otro"],
        }
