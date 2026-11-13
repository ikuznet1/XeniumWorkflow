// Plots panel: a click on a method note (ids ending in -note / -caveat) shows it in full,
// a second click folds it back to two lines (styles in sidebar.css).
document.addEventListener('click', function (ev) {
    var el = ev.target && ev.target.closest ? ev.target.closest('#umap-panel .tab-content [id$="-note"], #umap-panel .tab-content [id$="-caveat"]') : null;
    if (!el) return;
    if (window.getSelection && String(window.getSelection())) return;   // selecting text is not a toggle
    el.classList.toggle('note-open');
});
