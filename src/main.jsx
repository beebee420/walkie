import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import Walkie from './Walkie.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Walkie />
  </StrictMode>,
)
