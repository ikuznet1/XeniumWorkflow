"""Callbacks of session save / restore (session.py holds the logic and the allow-list).

The values of the controls live in the browser page. Three callbacks move them:

1. in the page, a click on a button of the Session dialog, an upload or the tick of the
   automatic save collects the value of every control of the allow-list into the store
   ``session-request`` ({action, controls, page; for an upload also the file: upload});
   after an upload the collector empties the upload component in the page
   (``dash_clientside.set_props``), so that choosing the same file again is a change;
2. ``session_action`` (the only server callback that does something) saves or restores;
   for a restore it returns the control values to set in ``session-apply``;
3. in the page, ``session-apply`` is written into the controls. Each control then fires
   the callbacks that depend on it, exactly as when the user changes it.
"""

import json

import dash
from dash import Input, Output, State, no_update

from . import session, app_layout


_SESSION_IDS = sorted(session._session_controls())
_SESSION_TRIGGERS = {"session-save-btn": "save", "session-download-btn": "download",
                     "session-restore-saved-btn": "restore-saved", "session-restore-last-btn": "restore-last",
                     "session-upload": "upload", "session-autosave": "auto"}

# 1. collect (browser)
app_layout.app.clientside_callback(
    """function() {
        var nu = window.dash_clientside.no_update;
        var trig = (window.dash_clientside.callback_context.triggered || [])[0];
        if (!trig || !trig.value) return nu;
        var actions = %s, ids = %s;
        var action = actions[trig.prop_id.split('.')[0]];
        if (!action) return nu;
        var first = arguments.length - ids.length, controls = {};
        for (var i = 0; i < ids.length; i++) {
            var v = arguments[first + i];
            controls[ids[i]] = (v === undefined) ? null : v;
        }
        // one id per page load: the automatic save tells a page that was only (re)opened
        // from one that goes on (session._session_autosave)
        if (!window.__xeSessionPage) {
            window.__xeSessionPage = Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
        }
        var req = {action: action, t: Date.now(), page: window.__xeSessionPage, controls: controls};
        if (action === 'upload') {
            // The file travels in the request, and the upload component is emptied here, in
            // the page - not by an Output of the server callback (an Input on a property
            // that a callback downstream writes is a dependency cycle), and not left as it
            // is (the same file chosen again would then change nothing and fire nothing).
            // The null this writes comes back as a trigger without a value: ignored above.
            req.upload = trig.value;
            window.dash_clientside.set_props('session-upload',
                                             {contents: null, filename: null, last_modified: null});
        }
        return req;
    }""" % (json.dumps(_SESSION_TRIGGERS), json.dumps(_SESSION_IDS)),
    Output("session-request", "data"),
    Input("session-save-btn", "n_clicks"),
    Input("session-download-btn", "n_clicks"),
    Input("session-restore-saved-btn", "n_clicks"),
    Input("session-restore-last-btn", "n_clicks"),
    # contents is the trigger (it changes with every file chosen, once the collector has
    # emptied it); no callback has it as an Output - see the collector.
    Input("session-upload", "contents"),
    Input("session-autosave", "n_intervals"),
    [State(cid, "value") for cid in _SESSION_IDS],
    prevent_initial_call=True,
)

# 3. apply (browser): only the controls the server names are written
app_layout.app.clientside_callback(
    """function(data) {
        var nu = window.dash_clientside.no_update, ids = %s;
        var vals = (data && data.controls) || {};
        return ids.map(function(id) {
            return Object.prototype.hasOwnProperty.call(vals, id) ? vals[id] : nu;
        });
    }""" % json.dumps(_SESSION_IDS),
    [Output(cid, "value", allow_duplicate=True) for cid in _SESSION_IDS],
    Input("session-apply", "data"),
    prevent_initial_call=True,
)


@app_layout.app.callback(
    Output("session-modal", "is_open"),
    Output("session-slots", "children"),
    Input("session-open-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_session_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return True, session._session_slots_text()


@app_layout.app.callback(
    Output("session-modal", "is_open", allow_duplicate=True),
    Input("session-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_session_modal(n_clicks):
    return False


# 2. act (server)
@app_layout.app.callback(
    Output("session-status", "children"),
    Output("session-slots", "children", allow_duplicate=True),
    Output("session-apply", "data"),
    Output("session-download", "data"),
    Output("session-autosave-note", "data"),
    Output("design-version", "data", allow_duplicate=True),
    Output("roi-done", "data", allow_duplicate=True),
    Output("geneset-version", "data", allow_duplicate=True),
    Output("de-groups-version", "data", allow_duplicate=True),
    Output("extra-datasets-version", "data", allow_duplicate=True),
    Input("session-request", "data"),
    State("design-version", "data"),
    State("roi-done", "data"),
    State("geneset-version", "data"),
    State("de-groups-version", "data"),
    State("extra-datasets-version", "data"),
    prevent_initial_call=True,
)
def session_action(request, design_v=None, roi_v=None, geneset_v=None, groups_v=None, extra_v=None):
    """Save, download or restore, as the page asked in ``session-request`` (an uploaded
    file is part of the request: request["upload"], the value of the upload component)."""
    nu = no_update
    quiet = (nu,) * 10
    if not isinstance(request, dict):
        raise dash.exceptions.PreventUpdate
    action = request.get("action")
    controls = request.get("controls") if isinstance(request.get("controls"), dict) else {}
    if action == "auto":
        done = session._session_autosave(controls, request.get("page"))
        return quiet[:4] + (done or nu,) + quiet[5:]
    try:
        if action in ("save", "download"):
            doc, msg = session._session_save(controls, "saved" if action == "save" else "download")
            download = nu
            if action == "download":
                stamp = doc["saved_at"][:19].replace(":", "").replace("-", "")
                download = {"content": session._session_text(doc), "filename": f"xenium_session_{stamp}.json",
                            "type": "application/json"}
            return (msg, session._session_slots_text(), nu, download) + quiet[4:]
        if action == "upload":
            report = session._session_restore_text(session._session_upload_bytes(request.get("upload")), controls,
                                                   "uploaded file")
        elif action in ("restore-saved", "restore-last"):
            report = session._session_restore_slot(action.split("-", 1)[1], controls)
        else:
            raise dash.exceptions.PreventUpdate
    except session._SessionError as exc:
        return (f"Not done. {exc}",) + quiet[1:]
    except dash.exceptions.PreventUpdate:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return (f"Error: {type(exc).__name__}: {exc}",) + quiet[1:]
    bump = lambda name, cur: (cur or 0) + 1 if name in report["bump"] else nu
    apply_ = {"t": request.get("t"), "controls": report["controls"]} if report["ok"] and report["controls"] else nu
    return (report["text"], session._session_slots_text(), apply_, nu, nu,
            bump("design-version", design_v), bump("roi-done", roi_v), bump("geneset-version", geneset_v),
            bump("de-groups-version", groups_v), bump("extra-datasets-version", extra_v))
