"use client";

import { Suspense, useLayoutEffect, useRef } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { Bounds, Grid, OrbitControls } from "@react-three/drei";
import { STLLoader } from "three-stdlib";
import type { BufferGeometry, Mesh } from "three";

function Model({ url }: { url: string }) {
  const geometry = useLoader(STLLoader, url) as BufferGeometry;
  const ref = useRef<Mesh>(null);

  useLayoutEffect(() => {
    // STL files carry no vertex normals → flat/again shading without this.
    geometry.computeVertexNormals();
    geometry.center();
  }, [geometry]);

  return (
    <mesh ref={ref} geometry={geometry} castShadow receiveShadow>
      <meshStandardMaterial color="#aebccd" metalness={0.15} roughness={0.5} />
    </mesh>
  );
}

export default function StlScene({ url }: { url: string }) {
  return (
    <Canvas
      shadows
      dpr={[1, 2]}
      camera={{ position: [60, 45, 60], fov: 42, near: 0.1, far: 5000 }}
      style={{ background: "transparent" }}
    >
      <hemisphereLight args={["#cdd8e6", "#0a0d12", 0.7]} />
      <directionalLight
        position={[40, 70, 30]}
        intensity={1.5}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[-50, 20, -30]} intensity={0.4} color="#3fd9c9" />

      <Suspense fallback={null}>
        <Bounds fit clip observe margin={1.25}>
          <Model url={url} />
        </Bounds>
      </Suspense>

      <Grid
        position={[0, -0.01, 0]}
        infiniteGrid
        cellSize={5}
        cellThickness={0.5}
        sectionSize={25}
        sectionThickness={1}
        cellColor="#222b38"
        sectionColor="#3fd9c9"
        fadeDistance={420}
        fadeStrength={2}
      />

      <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
    </Canvas>
  );
}
