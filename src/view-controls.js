// Availability follows the rendered view, not just the selected blur mode. Keep values when
// disabling controls so switching back restores the user's settings.
export function viewControls({ ready, source, blur, mode, layers, sizeBlur, occlusion, showLayers }) {
  const depthAvailable = ['world', 'video', 'shapes'].includes(source);
  const filtering = ready && blur;
  const depth = filtering && mode === 'depth' && depthAvailable;
  const varying = filtering && (depth || mode === 'tiltShift');
  const layered = depth && layers;
  const layerViews = layered ? ['0', '1', '2', '3', '5'] : ['0', '4'];
  if (depth) layerViews.push('6');
  const view = filtering && layerViews.includes(String(showLayers)) ? String(showLayers) : '0';
  const depthView = view === '6';
  const playback = ready && ['world', 'video'].includes(source);
  // Only videos have a duration and an end.
  const video = ready && source === 'video';
  return {
    depthAvailable,
    layerViews,
    view,
    disabled: {
      blurOn: !ready, blurMode: !filtering, radius: !filtering || depthView, focus: !depth || depthView,
      // Keep uniform kernel-weight diagnostics reachable inside this fieldset.
      varyingControls: !filtering, radiusInterpolation: !varying || depthView, groupTerms: !varying || depthView,
      layers: !depth || depthView, showLayers: !filtering,
      // Tilt-shift radii are constant along each row, so their horizontal average is unchanged.
      sizeBlur: !depth || depthView, sizeBrightness: !depth || !sizeBlur || depthView,
      occlusion: !depth || depthView, occlusionLevels: !depth || !occlusion || depthView,
      shapesScale: !ready || source !== 'shapes',
      playPause: !playback, stepBack: !playback, stepForward: !playback, sceneTime: !playback,
      toStart: !playback, playbackRate: !playback, seek: !video, loop: !video, fullscreen: !ready,
      precision: !filtering || depthView, follow: !filtering || depthView,
      exposure: !ready || depthView, saveFrame: !ready, copyView: !ready,
    },
  };
}