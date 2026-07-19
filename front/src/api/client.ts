const BASE_URL = '/api/v1'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
  })

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(errorBody.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

async function fetchRaw(url: string, options?: RequestInit): Promise<Response> {
  const response = await fetch(`${BASE_URL}${url}`, options)
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(errorBody.detail || `HTTP ${response.status}`)
  }
  return response
}

export { request, fetchRaw }
