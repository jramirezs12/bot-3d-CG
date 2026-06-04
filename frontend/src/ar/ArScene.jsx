import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

const ArScene = forwardRef(function ArScene({ facingMode = 'user' }, ref) {
  const containerRef = useRef(null)
  const videoRef     = useRef(null)

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = videoRef.current
      if (!video) return null
      const w = video.videoWidth || 640
      const h = video.videoHeight || 480
      if (!w || !h) return null
      const MAX = 640, scale = Math.min(MAX / w, MAX / h, 1)
      const cw = Math.round(w * scale), ch = Math.round(h * scale)
      try {
        const canvas = document.createElement('canvas')
        canvas.width = cw; canvas.height = ch
        canvas.getContext('2d').drawImage(video, 0, 0, cw, ch)
        const url = canvas.toDataURL('image/jpeg', 0.75)
        return url === 'data:,' ? null : url.split(',')[1]
      } catch { return null }
    }
  }))

  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let animFrameId, stream

    // ── Cámara ───────────────────────────────────────────────────────────
    const video = document.createElement('video')
    videoRef.current = video
    video.setAttribute('playsinline', ''); video.setAttribute('autoplay', ''); video.muted = true
    video.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:inherit;'
    container.appendChild(video)
    navigator.mediaDevices?.getUserMedia({ video: { facingMode }, audio: false })
      .then(s => { stream = s; video.srcObject = s; video.play() })
      .catch(() => { video.remove(); container.style.background = '#0b1220' })

    // ── Three.js ─────────────────────────────────────────────────────────
    const W = container.clientWidth || 640, H = container.clientHeight || 400
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(W, H); renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0); renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.shadowMap.enabled = true
    renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;border-radius:inherit;'
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(46, W / H, 0.01, 100)
    camera.position.set(0, 0.9, 3.2); camera.lookAt(0, 0.5, 0)

    scene.add(new THREE.AmbientLight(0xffffff, 0.9))
    const dir = new THREE.DirectionalLight(0xffffff, 1.4); dir.position.set(2, 4, 3); dir.castShadow = true; scene.add(dir)
    const fill = new THREE.DirectionalLight(0x6699ff, 0.5); fill.position.set(-2, 1, -1); scene.add(fill)

    // Sombra blob
    const shadowMesh = new THREE.Mesh(
      new THREE.CircleGeometry(0.38, 32),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.3 })
    )
    shadowMesh.rotation.x = -Math.PI / 2; shadowMesh.position.y = 0.001; scene.add(shadowMesh)

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.42, 0.014, 8, 64),
      new THREE.MeshStandardMaterial({ color: 0x60a5fa, emissive: 0x60a5fa, emissiveIntensity: 1.0, transparent: true, opacity: 0.7 })
    )
    ring.rotation.x = Math.PI / 2; ring.position.y = 0.002; scene.add(ring)

    const pCount = 60; const pPos = new Float32Array(pCount * 3)
    for (let i = 0; i < pCount; i++) { pPos[i*3]=(Math.random()-.5)*3.5; pPos[i*3+1]=Math.random()*3; pPos[i*3+2]=(Math.random()-.5)*2 }
    const pGeo = new THREE.BufferGeometry(); pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3))
    const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x93c5fd, size: 0.022, transparent: true, opacity: 0.6 }))
    scene.add(particles)

    // ── Bot ──────────────────────────────────────────────────────────────
    let botGroup=null, mouthMesh=null, armL=null, armR=null, antBall=null
    let isTalking=false, talkIntensity=0
    let botTargetPos = new THREE.Vector3(0, 0, 0)

    function buildBot() {
      const g = new THREE.Group(); g.scale.setScalar(0.72)
      const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.30,0.58,10,20), new THREE.MeshStandardMaterial({color:0x2563eb,roughness:0.35,metalness:0.35})); body.position.set(0,0.66,0); body.castShadow=true; g.add(body)
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.30,32,32), new THREE.MeshStandardMaterial({color:0x3b82f6,roughness:0.25,metalness:0.2})); head.position.set(0,1.38,0); head.castShadow=true; g.add(head)
      const eyeMat=new THREE.MeshStandardMaterial({color:0xffffff,emissive:0xbfdbfe,emissiveIntensity:0.6}), pupilMat=new THREE.MeshStandardMaterial({color:0x1e3a5f})
      ;[-0.11,0.11].forEach(x=>{ const e=new THREE.Mesh(new THREE.SphereGeometry(0.058,16,16),eyeMat); e.position.set(x,1.41,0.26); g.add(e); const p=new THREE.Mesh(new THREE.SphereGeometry(0.032,12,12),pupilMat); p.position.set(x,1.41,0.30); g.add(p) })
      const ms=new THREE.Shape(new THREE.EllipseCurve(0,0,0.09,0.025,0,Math.PI*2).getPoints(24)); mouthMesh=new THREE.Mesh(new THREE.ShapeGeometry(ms),new THREE.MeshStandardMaterial({color:0x1e3a5f,side:THREE.DoubleSide})); mouthMesh.position.set(0,1.27,0.285); mouthMesh.rotation.x=-0.15; g.add(mouthMesh)
      const sk=new THREE.Mesh(new THREE.CylinderGeometry(0.013,0.013,0.28,8),new THREE.MeshStandardMaterial({color:0x94a3b8})); sk.position.set(0,1.80,0); g.add(sk)
      antBall=new THREE.Mesh(new THREE.SphereGeometry(0.048,16,16),new THREE.MeshStandardMaterial({color:0xfbbf24,emissive:0xfbbf24,emissiveIntensity:1.0})); antBall.position.set(0,1.96,0); g.add(antBall)
      ;[[-1,'L'],[1,'R']].forEach(([side,label])=>{ const pv=new THREE.Group(); pv.position.set(side*0.36,0.90,0); pv.rotation.z=side*0.52; const arm=new THREE.Mesh(new THREE.CapsuleGeometry(0.078,0.34,8,12),new THREE.MeshStandardMaterial({color:0x1d4ed8,roughness:0.38,metalness:0.32})); arm.position.set(0,-0.20,0); arm.castShadow=true; pv.add(arm); const hand=new THREE.Mesh(new THREE.SphereGeometry(0.085,16,16),new THREE.MeshStandardMaterial({color:0x3b82f6,roughness:0.28})); hand.position.set(0,-0.40,0); pv.add(hand); g.add(pv); if(label==='L')armL=pv; else armR=pv })
      ;[-0.13,0.13].forEach(x=>{ const l=new THREE.Mesh(new THREE.CapsuleGeometry(0.075,0.22,8,10),new THREE.MeshStandardMaterial({color:0x1e40af,roughness:0.45})); l.position.set(x,0.14,0); g.add(l); const f=new THREE.Mesh(new THREE.SphereGeometry(0.085,14,14),new THREE.MeshStandardMaterial({color:0x1d4ed8})); f.position.set(x,0.02,0.05); g.add(f) })
      scene.add(g); botGroup=g; window.__AR_BOT_MODEL=g
    }

    const loader=new GLTFLoader(); let mixer=null
    loader.load('/bot.glb', gltf=>{ const m=gltf.scene; m.scale.setScalar(0.72); scene.add(m); botGroup=m; window.__AR_BOT_MODEL=m; if(gltf.animations?.length){mixer=new THREE.AnimationMixer(m);mixer.clipAction(gltf.animations[0]).play()} }, undefined, ()=>buildBot())

    // ── Tap-to-place ─────────────────────────────────────────────────────
    const raycaster=new THREE.Raycaster(), floorPlane=new THREE.Plane(new THREE.Vector3(0,1,0),0), tapTarget=new THREE.Vector3()
    function onTap(e) {
      const rect=renderer.domElement.getBoundingClientRect()
      const cx=e.touches?e.touches[0].clientX:e.clientX, cy=e.touches?e.touches[0].clientY:e.clientY
      const ndc=new THREE.Vector2(((cx-rect.left)/rect.width)*2-1,-((cy-rect.top)/rect.height)*2+1)
      raycaster.setFromCamera(ndc,camera)
      if(raycaster.ray.intersectPlane(floorPlane,tapTarget)) { botTargetPos.copy(tapTarget); hint.style.opacity='0' }
    }
    container.style.cursor='crosshair'
    container.addEventListener('pointerdown',onTap)
    container.addEventListener('touchstart',onTap,{passive:true})

    // ── Overlays ─────────────────────────────────────────────────────────
    const hint=document.createElement('div'); hint.style.cssText='position:absolute;bottom:52px;left:50%;transform:translateX(-50%);background:rgba(10,10,30,0.65);color:#93c5fd;padding:5px 14px;border-radius:20px;font-size:11px;font-family:-apple-system,Inter,sans-serif;pointer-events:none;border:1px solid rgba(96,165,250,0.2);backdrop-filter:blur(8px);white-space:nowrap;transition:opacity 0.5s;'; hint.textContent='Toca para colocar el bot'; container.appendChild(hint); setTimeout(()=>hint.style.opacity='0',3000)
    const indicator=document.createElement('div'); indicator.style.cssText='position:absolute;bottom:12px;left:50%;transform:translateX(-50%);background:rgba(10,10,30,0.72);color:#93c5fd;padding:5px 16px;border-radius:20px;font-size:12px;font-family:-apple-system,Inter,sans-serif;pointer-events:none;border:1px solid rgba(96,165,250,0.25);backdrop-filter:blur(8px);transition:color 0.3s;white-space:nowrap;'; indicator.textContent='🤖 Listo'; container.appendChild(indicator)

    window.__AR_BOT_SPEAK=text=>{ isTalking=true; indicator.textContent='🗣️ Respondiendo…'; indicator.style.color='#4ade80'; setTimeout(()=>{ isTalking=false; indicator.textContent='🤖 Listo'; indicator.style.color='#93c5fd' },Math.min((text?.split(' ').length||10)*380,9000)) }

    // ── Render loop ──────────────────────────────────────────────────────
    const clock=new THREE.Clock()
    const renderLoop=()=>{
      animFrameId=requestAnimationFrame(renderLoop)
      const t=clock.getElapsedTime()
      talkIntensity+=(( isTalking?1:0)-talkIntensity)*0.08
      if(botGroup&&!botFixed) { botGroup.position.lerp(botTargetPos,0.12); shadowMesh.position.x=botGroup.position.x; shadowMesh.position.z=botGroup.position.z; ring.position.x=botGroup.position.x; ring.position.z=botGroup.position.z }
      if(mixer) mixer.update(0.016)
      if(botGroup&&!mixer){
        const bobY=Math.sin(t*1.3)*0.055; botGroup.position.y=botTargetPos.y+bobY; botGroup.rotation.y=Math.sin(t*0.55)*0.10*(1-talkIntensity*0.5)
        shadowMesh.scale.setScalar(Math.max(0.1,1-bobY*3)); shadowMesh.material.opacity=0.3-bobY*0.08
        if(mouthMesh) mouthMesh.scale.y=THREE.MathUtils.lerp(mouthMesh.scale.y,talkIntensity>0.05?1+Math.abs(Math.sin(t*10))*3.5*talkIntensity:0.4,0.25)
        if(armL&&armR){ const raise=talkIntensity*0.9,swing=Math.sin(t*7)*0.35*talkIntensity,idle=Math.sin(t*1.1)*0.06; armL.rotation.z=0.52-raise+idle+swing; armR.rotation.z=-0.52+raise-idle-swing }
        if(antBall) antBall.material.emissiveIntensity=0.8+talkIntensity*1.5+Math.sin(t*(isTalking?12:3))*0.3
      }
      ring.material.emissiveIntensity=0.8+Math.sin(t*2.4)*0.4+talkIntensity*0.6; ring.scale.setScalar(1+Math.sin(t*2.4)*0.025)
      const pa=particles.geometry.attributes.position.array; for(let i=0;i<pCount;i++){pa[i*3+1]+=0.003+talkIntensity*0.004;if(pa[i*3+1]>3)pa[i*3+1]=0}; particles.geometry.attributes.position.needsUpdate=true
      renderer.render(scene,camera)
    }
    let botFixed=false
    renderLoop()

    const ro=new ResizeObserver(()=>{ const w=container.clientWidth,h=container.clientHeight; camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h) }); ro.observe(container)

    return()=>{ cancelAnimationFrame(animFrameId); ro.disconnect(); stream?.getTracks().forEach(t=>t.stop()); renderer.dispose(); video.remove(); renderer.domElement.remove(); indicator.remove(); hint.remove(); container.removeEventListener('pointerdown',onTap); container.removeEventListener('touchstart',onTap); container.style.cursor=''; delete window.__AR_BOT_MODEL; delete window.__AR_BOT_SPEAK }
  }, [facingMode])

  return <div ref={containerRef} style={{position:'relative',width:'100%',height:'100%',borderRadius:'inherit',overflow:'hidden',background:'#0b1220'}} />
})

export default ArScene
