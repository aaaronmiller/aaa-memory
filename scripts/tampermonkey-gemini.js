// ==UserScript==
// @name         Clawdi Memory Capture — Gemini
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  Capture Gemini conversations to aaa-memory
// @author       You
// @match        https://gemini.google.com/*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';

    const OUTPUT_DIR = '/home/misscheta/knowledge/raw/web/gemini/';

    console.log('[Clawdi] Gemini capture loaded');

    // Gemini uses <json-model-response> elements or network intercepts
    // Observe DOM for message nodes
    const observer = new MutationObserver(() => {
        captureFromDOM();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    function captureFromDOM() {
        // Extract conversation turns from Gemini UI
        const messages = [];
        document.querySelectorAll('model-response, conversation-turn').forEach(el => {
            const role = el.getAttribute('role') || (el.querySelector('[role="img"]') ? 'model' : 'user');
            const content = el.textContent?.trim() || '';
            if (content) {
                messages.push({ role, content });
            }
        });

        if (messages.length === 0) return;

        const turn = {
            timestamp: new Date().toISOString(),
            platform: 'gemini',
            url: window.location.href,
            messages
        };

        saveTurn(turn);
    }

    function saveTurn(turn) {
        const filename = `${OUTPUT_DIR}${Date.now()}-${Math.random().toString(36).slice(2,8)}.jsonl`;
        const line = JSON.stringify(turn) + '\n';

        if (typeof GM_xmlhttpRequest !== 'undefined') {
            GM_xmlhttpRequest({
                method: 'POST',
                url: 'file://' + filename,
                data: line,
                headers: { 'Content-Type': 'application/json' }
            }).catch(() => { saveToLocalStorage(turn); });
        } else {
            saveToLocalStorage(turn);
        }
    }

    function saveToLocalStorage(turn) {
        try {
            const key = 'clawdi_capture_' + Date.now();
            localStorage.setItem(key, JSON.stringify(turn));
        } catch (e) {}
    }

    console.log('[Clawdi] Gemini capture initialized');
})();
