import type { HealthResponse } from '../types'

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch('/health')
  if (!response.ok) {
    throw new Error('服务不可用')
  }
  return response.json()
}
