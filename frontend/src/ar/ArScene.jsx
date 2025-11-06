import React, { useEffect, useRef } from 'react'

// WebAR con MindAR (CDN) + Three.js. El modelo se ancla al target #0.

export default function ArScene(){
  const containerRef = useRef(null)

  useEffect(() => {
    let mindarThree
    let stopped = false

    async function ensureScripts() {
      if (!window.MINDAR) {
        await new Promise((res) => {
          const s = document.createElement('script')
          // Librería que integra MindAR + Three
          s.src = 'https://cdn.jsdelivr.net/npm/mind-ar@1.2.2/dist/mindar-image-three.prod.js'
          s.onload = res
          document.head.appendChild(s)
        })
      }
      if (!window.THREE) {
        await new Promise((res) => {
          const s = document.createElement('script')
          s.src = 'https://unpkg.com/three@0.156.0/build/three.min.js'
          s.onload = res
          document.head.appendChild(s)
        })
      }
    }

    async function init(){
      await ensureScripts()
      if (stopped) return

      const { IMAGE } = window.MINDAR
      const mindarThreeModule = IMAGE
      mindarThree = new mindarThreeModule.MindARThree({
        container: containerRef.current,
        imageTargetSrc: '/target.mind',
        uiScanning: true,
        uiLoading: 'cargando...'
      })

      const { renderer, scene, camera } = mindarThree
      const anchor = mindarThree.addAnchor(0)

      // Cargar GLTFLoader de three/examples vía import dinámico (usa dependencia local de three)
      const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
      const loader = new GLTFLoader()

      loader.load(
        '/bot.glb',
        (gltf) => {
          const model = gltf.scene
          model.scale.set(0.05, 0.05, 0.05)
          model.position.set(0, 0, 0)
          anchor.group.add(model)
          window.__AR_BOT_MODEL = model
        },
        undefined,
        (err) => {
          console.error('Error cargando GLB:', err)
        }
      )

      await mindarThree.start()
      renderer.setAnimationLoop(() => {
        renderer.render(scene, camera)
      })
    }

    init()

    return () => {
      stopped = true
      if (mindarThree) {
        const { renderer } = mindarThree
        try { mindarThree.stop() } catch {}
        if (renderer) renderer.setAnimationLoop(null)
      }
    }
  }, [])

  return <div ref={containerRef} style={{ width: '100%', height: '60vh' }} />
}