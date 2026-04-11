/**
 * NIOR-AI Public Annotation API (Section IV, Listing 2)
 *
 * Exposes a singleton NIOR_AI object wrapping ShadowRenderer so that
 * callers do not need to manage the renderer lifecycle directly.
 *
 * Usage (from content scripts or injected scripts):
 *
 *   import { NIOR_AI } from '../annotation-api/index.js';
 *
 *   const h = NIOR_AI.annotate(element, {
 *     type:     'bbox',        // 'bbox' | 'highlight' | 'label' | 'heatmap'
 *     color:    '#FF4040',
 *     label:    'navigation',
 *     labelPos: 'top-left'    // 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
 *   });
 *
 *   NIOR_AI.update(h, { color: '#00CC44' });
 *   NIOR_AI.remove(h);
 *   NIOR_AI.batch(() => elements.forEach(el => NIOR_AI.annotate(el, cfg)));
 *   NIOR_AI.clear();
 */

'use strict';

import { ShadowRenderer } from '../shadow-renderer/renderer.js';

let _renderer = null;

function getRenderer() {
  if (!_renderer) _renderer = new ShadowRenderer();
  return _renderer;
}

export const NIOR_AI = {
  /**
   * Annotate a DOM element.
   *
   * @param {Element} element
   * @param {{ type?: 'bbox'|'highlight'|'label'|'heatmap',
   *            color?: string, label?: string, labelPos?: string,
   *            fill?: boolean }} config
   * @returns {number} Opaque annotation handle
   */
  annotate(element, config = {}) {
    return getRenderer().annotate(element, config);
  },

  /**
   * Update the config of an existing annotation.
   *
   * @param {number} handle
   * @param {object} config  Partial config — only provided keys are updated
   */
  update(handle, config) {
    getRenderer().update(handle, config);
  },

  /**
   * Remove a single annotation by handle.
   *
   * @param {number} handle
   */
  remove(handle) {
    getRenderer().remove(handle);
  },

  /**
   * Batch multiple annotate/update/remove calls into a single rAF flush.
   * Prevents partial-frame flicker when classifying many elements at once.
   *
   * @param {() => void} fn  Callback containing annotation calls
   */
  batch(fn) {
    getRenderer().batch(fn);
  },

  /**
   * Remove all annotations and clear the canvas.
   */
  clear() {
    getRenderer().clear();
  },

  /**
   * Tear down the renderer and release all resources.
   * Call when the extension is deactivated.
   */
  destroy() {
    if (_renderer) {
      _renderer.destroy();
      _renderer = null;
    }
  }
};
