import assert from 'node:assert/strict';
import { viewControls } from '../src/view-controls.js';

const defaults = { ready: true, source: 'video', blur: true, mode: 'depth', layers: false,
  sizeBlur: true, occlusion: true, showLayers: '0' };
const controls = overrides => viewControls({ ...defaults, ...overrides });

export function testViewControls() {
  for (const source of ['video', 'world', 'shapes', 'image', 'procedural']) {
    for (const mode of ['uniform', 'tiltShift', 'depth']) {
      for (const blur of [false, true]) for (const layers of [false, true]) {
        const c = controls({ source, mode, blur, layers });
        const depth = blur && mode === 'depth' && ['video', 'world', 'shapes'].includes(source);
        assert.equal(!c.disabled.focus, depth);
        assert.equal(!c.disabled.layers, depth);
        assert.equal(!c.disabled.sizeBrightness, depth);
        assert.equal(!c.disabled.radiusInterpolation, blur && (depth || mode === 'tiltShift'));
        assert.equal(!c.disabled.groupTerms, blur && (depth || mode === 'tiltShift'));
        assert.equal(!c.disabled.playPause, ['video', 'world'].includes(source));
        assert.equal(!c.disabled.playbackRate, ['video', 'world'].includes(source));
        assert.equal(!c.disabled.seek, source === 'video');
        assert.equal(!c.disabled.loop, source === 'video');
        assert.equal(c.disabled.fullscreen, false);
        assert.equal(!c.disabled.shapesScale, source === 'shapes');
        assert.equal(!c.disabled.precision, blur);
        assert.equal(c.layerViews.includes('1'), depth && layers);
        assert.equal(c.layerViews.includes('4'), !(depth && layers));
        assert.equal(c.layerViews.includes('6'), depth);
      }
    }
  }
  assert.equal(controls({ sizeBlur: false }).disabled.sizeBrightness, true);
  assert.equal(controls({ occlusion: false }).disabled.occlusionLevels, true);
  const depthView = controls({ showLayers: '6' });
  assert.equal(depthView.view, '6');
  for (const id of ['radius', 'focus', 'layers', 'sizeBlur', 'sizeBrightness', 'occlusion',
    'occlusionLevels', 'precision', 'follow', 'exposure', 'radiusInterpolation']) {
    assert.equal(depthView.disabled[id], true, `depth view: ${id}`);
  }
  assert.equal(depthView.disabled.showLayers, false);
  assert.equal(controls({ showLayers: '6', layers: true }).view, '6');
  assert.equal(controls({ showLayers: '6', mode: 'uniform' }).view, '0');
  assert.equal(controls({ mode: 'uniform' }).disabled.varyingControls, false);
  assert.equal(controls({ showLayers: '3' }).view, '0');
  assert.equal(controls({ showLayers: '3', layers: true }).view, '3');
  assert.equal(controls({ showLayers: '4', mode: 'uniform' }).view, '4');
  assert.equal(controls({ showLayers: '4', layers: true }).view, '0');
  assert.equal(controls({ showLayers: '2', layers: true, blur: false }).view, '0');
  assert.ok(Object.values(controls({ ready: false }).disabled).every(Boolean));
}