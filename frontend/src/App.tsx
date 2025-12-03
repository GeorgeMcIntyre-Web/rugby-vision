import React, { useState } from 'react'
import './App.css'

interface DecisionResult {
  isForward: boolean
  confidence: number
  explanation: string
}

function App(): React.ReactElement {
  const [result, setResult] = useState<DecisionResult | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyseClick = async (): Promise<void> => {
    setLoading(true)
    setError(null)
    
    try {
      // Guard clause: validate before proceeding
      if (!window.fetch) {
        throw new Error('Fetch API not available')
      }

      const response = await fetch('/api/clip/analyse-pass', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          clipId: 'demo-clip-1',
          cameras: ['cam1', 'cam2', 'cam3'],
          startTime: 0,
          endTime: 5,
        }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json() as DecisionResult
      setResult(data)
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Rugby Vision</h1>
        <p>Multi-Camera 3D Forward Pass Detection</p>
      </header>

      <main className="app-main">
        <section className="video-section">
          <div className="video-placeholder">
            <p>Video Player (Multi-camera view)</p>
            <p className="placeholder-text">Phase 1: POC - Video player integration pending</p>
          </div>
        </section>

        <section className="control-section">
          <button 
            onClick={handleAnalyseClick}
            disabled={loading}
            className="analyse-button"
          >
            {loading ? 'Analysing...' : 'Analyse Pass'}
          </button>
        </section>

        <section className="result-section">
          {error && (
            <div className="error-box">
              <strong>Error:</strong> {error}
            </div>
          )}

          {result && (
            <div className="result-box">
              <div className={`decision-indicator ${result.isForward ? 'forward' : 'not-forward'}`}>
                {result.isForward ? '🔴 FORWARD PASS' : '🟢 NOT FORWARD'}
              </div>
              <div className="confidence">
                <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
              </div>
              <div className="explanation">
                <strong>Explanation:</strong>
                <p>{result.explanation}</p>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
