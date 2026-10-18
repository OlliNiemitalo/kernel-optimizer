// Availability follows the rendered view, not just the selected blur mode. Keep values when
// disabling controls so switching back restores the user's settings.
export function viewControls({ ready, source, blur, mode, layers, sizeBlur, occlusion, showLayers }) {
  const depthAvailable = ['world', 'video', 'shapes'].includes(source);
  const filtering = ready && blur;
  const depth = filtering && mode === 'depth' && depthAvailable;
  const varying = filtering && (depth || mode === 'tiltShift');
  const layered = depth && layers;
  const layerViews = layered ? ['0', '1', '2', '3', '5'] : ['0', '4'];
  const view = filtering && layerViews.includes(String(showLayers)) ? String(showLayers) : '0';
  const playback = ready && ['world', 'video'].includes(source);
  return {
    depthAvailable,
    layerViews,
    view,
    disabled: {
      blurOn: !ready, blurMode: !filtering, radius: !filtering, focus: !depth,
      varyingControls: !varying, radiusInterpolation: !varying,
      layers: !depth, showLayers: !filtering,
      // Tilt-shift radii are constant along each row, so their horizontal average is unchanged.
      sizeBlur: !depth, sizeBrightness: !depth || !sizeBlur,
      occlusion: !depth, occlusionLevels: !depth || !occlusion,
      shapesScale: !ready || source !== 'shapes',
      boost: !ready || !['video', 'image'].includes(source),
      playPause: !playback, stepBack: !playback, stepForward: !playback, sceneTime: !playback,
      precision: !filtering, follow: !filtering,
      exposure: !ready, tonemap: !ready, saveFrame: !ready, copyView: !ready,
    },
  };
}