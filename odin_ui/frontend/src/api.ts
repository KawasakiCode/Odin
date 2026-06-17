import type { AnalyzeResult } from './types'

export async function analyze(file: File, sex: string): Promise<AnalyzeResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('sex', sex)

  const res = await fetch('/api/analyze', { method: 'POST', body: form })
  if (!res.ok) {
    let message = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) message = body.detail
    } catch {
      /* non-JSON error body — keep the generic message */
    }
    throw new Error(message)
  }
  return res.json() as Promise<AnalyzeResult>
}
