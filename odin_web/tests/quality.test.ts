// Head-pose extraction and the photo-quality thresholds.
import { describe, expect, it } from 'vitest'
import { photoWarnings, poseFromMatrix } from '../src/core/quality'

const R = (deg: number) => (deg * Math.PI) / 180

/** 4x4 row-major transform R = Rz(roll) Ry(yaw) Rx(pitch), translated to z = -30. */
function matrix(yaw: number, pitch: number, roll: number): number[] {
  const [cy, sy, cp, sp, cr, sr] = [Math.cos(R(yaw)), Math.sin(R(yaw)), Math.cos(R(pitch)),
    Math.sin(R(pitch)), Math.cos(R(roll)), Math.sin(R(roll))]
  const m = [
    [cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp],
    [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
    [-sy, cy * sp, cy * cp],
  ]
  return [...m[0], 0.4, ...m[1], -0.9, ...m[2], -30, 0, 0, 0, 1]
}
const transpose = (m: number[]) => m.map((_, k) => m[4 * (k % 4) + Math.floor(k / 4)])

describe('poseFromMatrix', () => {
  it.each([[0, 0, 0], [16, 3, 0.5], [-12, -7, 9], [4, 21, -15]])('recovers yaw %d, pitch %d, roll %d', (yaw, pitch, roll) => {
    for (const m of [matrix(yaw, pitch, roll), transpose(matrix(yaw, pitch, roll))]) {   // both layouts
      const p = poseFromMatrix(m)
      expect(p.yaw).toBeCloseTo(yaw, 9)
      expect(p.pitch).toBeCloseTo(pitch, 9)
      expect(p.roll).toBeCloseTo(roll, 9)
    }
  })
})

describe('photoWarnings', () => {
  const neutral = { smile: 0.05, jawOpen: 0.01 }
  const typical = { yaw: 1, pitch: 5, roll: 0 }

  it('says nothing for a frontal, neutral photo', () => {
    expect(photoWarnings(typical, neutral)).toEqual([])
  })

  it('flags each pose problem just past its threshold, not before', () => {
    expect(photoWarnings({ ...typical, yaw: 9.9 }, neutral)).toEqual([])
    expect(photoWarnings({ ...typical, yaw: -10.1 }, neutral)[0].text).toMatch(/turned 10°/)
    expect(photoWarnings({ ...typical, roll: 8.1 }, neutral)[0].text).toMatch(/tilted 8° sideways/)
    expect(photoWarnings({ ...typical, pitch: 23 }, neutral)[0].text).toMatch(/up or down/)
    expect(photoWarnings({ ...typical, pitch: -11 }, neutral)).toHaveLength(1)
  })

  it('warns on an open mouth but only notes a smile', () => {
    expect(photoWarnings(typical, { smile: 0.1, jawOpen: 0.5 })).toMatchObject([{ level: 'warn' }])
    expect(photoWarnings(typical, { smile: 0.9, jawOpen: 0.05 })).toMatchObject([{ level: 'info' }])
  })

  it('tolerates missing pose/expression', () => {
    expect(photoWarnings(null, null)).toEqual([])
  })
})
