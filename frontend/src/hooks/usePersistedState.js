import { useState, useEffect } from 'react'

// In-memory cache so remount is instant.
const _mem = {}

export default function usePersistedState(key, initial) {
  const [val, setVal] = useState(() => {
    if (_mem[key] !== undefined) return _mem[key]
    try {
      const raw = sessionStorage.getItem(key)
      if (raw !== null) return JSON.parse(raw)
    } catch {}
    return initial
  })

  useEffect(() => {
    _mem[key] = val
    try {
      sessionStorage.setItem(key, JSON.stringify(val))
    } catch {}
  }, [key, val])

  return [val, setVal]
}
