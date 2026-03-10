/**
 * REPL enhancements: Tab autocomplete + Up/Down arrow history
 * Attaches to the #repl-input element after Dash renders it.
 */
(function () {
  'use strict';

  // ── Completions list ──────────────────────────────────────────────────────
  // Names available in the app module namespace (callable from the REPL).
  const REPL_NAMES = [
    // Cell subsetting / gene utilities
    'subset', 'unsubset', 'get_genes', 'run_spage',
    // SpatialData / sopa
    'to_spatialdata', 'segment_tissue_roi', 'create_patches', 'clear_sdata_cache',
    // Segmentation
    '_run_baysor', '_run_proseg',
    // Annotation
    '_run_seurat_annotation', '_run_celltypist',
    // SpaGE
    '_run_spage_imputation',
    // State dicts
    'DATA', '_spage_state', '_annot_state', '_baysor_state', '_proseg_state', '_sdata_state',
    '_subset_version', '_sdata_version',
    // Data fields
    'DATA["df"]', 'DATA["gene_names"]', 'DATA["metadata"]', 'DATA["data_dir"]',
    'DATA["sdata"]', 'DATA["cluster_methods"]',
    // Common Python builtins useful in the REPL
    'len', 'list', 'print', 'sorted', 'set', 'dict', 'type', 'dir', 'vars', 'help',
    'pd', 'np', 'os', 'json',
  ];

  // ── State ─────────────────────────────────────────────────────────────────
  const cmdHistory = [];
  let historyIdx   = -1;   // -1 = not navigating
  let savedInput   = '';   // what user typed before pressing Up

  let completions    = [];
  let completionIdx  = -1;
  let completionBase = '';  // value up to start of the token being completed

  // ── Helpers ───────────────────────────────────────────────────────────────
  function triggerDashInput(input) {
    // Tell Dash's React that the input value changed
    const nativeInput = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    nativeInput.set.call(input, input.value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function getPrefix(value) {
    // Match the last identifier token (may contain dots)
    const m = value.match(/([\w.[\]'"_]*)$/);
    return m ? m[1] : '';
  }

  // ── Keydown handler ───────────────────────────────────────────────────────
  function onKeyDown(e) {
    const input = e.target;

    if (e.key === 'Tab') {
      e.preventDefault();
      const value  = input.value;
      const prefix = getPrefix(value);
      const base   = value.slice(0, value.length - prefix.length);

      if (completions.length === 0 || base !== completionBase || !completions[0].startsWith(prefix)) {
        // Start a new completion cycle
        completionBase = base;
        completions    = REPL_NAMES.filter(n => n.startsWith(prefix));
        completionIdx  = 0;
      } else {
        completionIdx  = (completionIdx + 1) % completions.length;
      }

      if (completions.length > 0) {
        input.value = completionBase + completions[completionIdx];
        triggerDashInput(input);
      }
      return;
    }

    // Reset completions on any non-navigation key
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') {
      completions   = [];
      completionIdx = -1;
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (cmdHistory.length === 0) return;
      if (historyIdx === -1) {
        savedInput = input.value;
        historyIdx = cmdHistory.length - 1;
      } else if (historyIdx > 0) {
        historyIdx--;
      }
      input.value = cmdHistory[historyIdx];
      triggerDashInput(input);
      // Move cursor to end
      setTimeout(() => input.setSelectionRange(input.value.length, input.value.length), 0);
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIdx === -1) return;
      if (historyIdx < cmdHistory.length - 1) {
        historyIdx++;
        input.value = cmdHistory[historyIdx];
      } else {
        historyIdx  = -1;
        input.value = savedInput;
      }
      triggerDashInput(input);
      setTimeout(() => input.setSelectionRange(input.value.length, input.value.length), 0);
      return;
    }

    if (e.key === 'Enter') {
      const cmd = input.value.trim();
      if (cmd && (cmdHistory.length === 0 || cmdHistory[cmdHistory.length - 1] !== cmd)) {
        cmdHistory.push(cmd);
      }
      historyIdx = -1;
      savedInput = '';
    }
  }

  // ── Auto-scroll log ───────────────────────────────────────────────────────
  // Scrolls #server-log to the bottom whenever new content is added, unless
  // the user has manually scrolled up. Resumes auto-scroll when they return
  // to the bottom.

  let logEl        = null;
  let userScrolled = false;   // true while user is scrolled up

  function isAtBottom(el) {
    // Allow 4 px of slack so a sub-pixel difference doesn't lock auto-scroll
    return el.scrollHeight - el.scrollTop - el.clientHeight < 4;
  }

  function onLogScroll() {
    userScrolled = !isAtBottom(logEl);
  }

  function scrollToBottom() {
    if (!userScrolled && logEl) {
      logEl.scrollTop = logEl.scrollHeight;
    }
  }

  function attachLog() {
    const el = document.getElementById('server-log');
    if (!el || el === logEl) return false;
    logEl = el;
    userScrolled = false;

    logEl.removeEventListener('scroll', onLogScroll);
    logEl.addEventListener('scroll', onLogScroll);

    // Watch for content mutations (Dash replaces children on each poll)
    const obs = new MutationObserver(scrollToBottom);
    obs.observe(logEl, { childList: true, subtree: true, characterData: true });

    scrollToBottom();
    return true;
  }

  // ── Attach to #repl-input (wait for Dash to render it) ───────────────────
  function attach() {
    const el = document.getElementById('repl-input');
    if (!el) return false;
    el.removeEventListener('keydown', onKeyDown); // avoid double-attach on re-render
    el.addEventListener('keydown', onKeyDown);
    el.setAttribute('autocomplete', 'off');
    el.setAttribute('spellcheck', 'false');
    return true;
  }

  function waitAndAttach() {
    attach();
    attachLog();
    if (!document.getElementById('repl-input') || !document.getElementById('server-log')) {
      const obs = new MutationObserver(() => {
        const gotRepl = attach();
        const gotLog  = attachLog();
        if (gotRepl && gotLog) obs.disconnect();
      });
      obs.observe(document.body, { childList: true, subtree: true });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitAndAttach);
  } else {
    waitAndAttach();
  }

  // Re-attach after Dash hot-reloads (e.g. callback updates the component)
  window.addEventListener('_dash-layout-modified', waitAndAttach);
})();
