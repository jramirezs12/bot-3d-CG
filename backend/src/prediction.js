import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder'

let model = null

export async function loadModel() {
  if (model) return model
  model = await use.load()
  return model
}

export async function getEmbedding(text) {
  const m = await loadModel()
  const embeddings = await m.embed([text])
  const vector = Array.from(await embeddings.data())
  embeddings.dispose()
  return vector
}

export async function calculateSimilarity(text1, text2) {
  const m = await loadModel()
  const embeddings = await m.embed([text1, text2])
  const [e1, e2] = tf.split(embeddings, 2)
  const dotProduct = tf.sum(tf.mul(e1, e2))
  const norm1 = tf.norm(e1)
  const norm2 = tf.norm(e2)
  const similarity = dotProduct.div(norm1.mul(norm2))
  const score = (await similarity.data())[0]
  tf.dispose([embeddings, e1, e2, dotProduct, norm1, norm2, similarity])
  return score
}

const INTENT_EXAMPLES = {
  greeting: ['hola', 'buenos días', 'buenas tardes', 'qué tal', 'hey', 'saludos'],
  farewell: ['adiós', 'hasta luego', 'nos vemos', 'chao', 'bye', 'hasta pronto'],
  help: ['ayuda', 'necesito ayuda', 'puedes ayudarme', 'qué puedes hacer', 'cómo funciona'],
  question: ['qué es', 'cómo', 'cuándo', 'dónde', 'por qué', 'cuál', 'quién'],
  thanks: ['gracias', 'muchas gracias', 'te lo agradezco', 'genial gracias'],
}

export async function classifyIntent(userText) {
  const m = await loadModel()
  const allExamples = Object.entries(INTENT_EXAMPLES).flatMap(([intent, examples]) =>
    examples.map(ex => ({ intent, text: ex }))
  )
  const texts = [userText, ...allExamples.map(e => e.text)]
  const embeddings = await m.embed(texts)
  const [userEmb, ...exampleEmbs] = tf.split(embeddings, texts.length)

  let bestIntent = 'unknown'
  let bestScore = -1
  const scores = {}
  const normUser = tf.norm(userEmb)

  for (let i = 0; i < exampleEmbs.length; i++) {
    const dot = tf.sum(tf.mul(userEmb, exampleEmbs[i]))
    const norm2 = tf.norm(exampleEmbs[i])
    const sim = dot.div(normUser.mul(norm2))
    const score = (await sim.data())[0]
    tf.dispose([dot, norm2, sim])
    const intent = allExamples[i].intent
    if (!(intent in scores) || score > scores[intent]) {
      scores[intent] = score
    }
    if (score > bestScore) {
      bestScore = score
      bestIntent = intent
    }
  }

  tf.dispose([embeddings, userEmb, normUser, ...exampleEmbs])
  return { intent: bestIntent, confidence: bestScore, scores }
}

const POSITIVE_WORDS = ['bien', 'genial', 'gracias', 'excelente', 'fantástico', 'perfecto', 'feliz', 'amor', 'bueno', 'increíble']
const NEGATIVE_WORDS = ['mal', 'error', 'problema', 'falla', 'no funciona', 'terrible', 'horrible', 'odio', 'triste', 'pésimo']

export function analyzeSentiment(text) {
  const lower = text.toLowerCase()
  let score = 0
  for (const w of POSITIVE_WORDS) if (lower.includes(w)) score++
  for (const w of NEGATIVE_WORDS) if (lower.includes(w)) score--
  let sentiment = 'neutral'
  if (score > 0) sentiment = 'positive'
  else if (score < 0) sentiment = 'negative'
  return { sentiment, score }
}
