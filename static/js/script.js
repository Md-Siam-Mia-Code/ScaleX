// No changes to JS file for this specific request (fixing .image-display-container visibility).
// The fix was purely in CSS by removing the conflicting `display: flex;` from `.image-display-container`.
// Re-pasting the previous full JS for completeness.

document.addEventListener('DOMContentLoaded', function () {
     // --- DOM Element Selectors ---
     const scalexForm = document.getElementById('scalexForm');
     const inputFile = document.getElementById('inputFile');
     const originalImageInfo = document.getElementById('originalImageInfo');
     const upscaledImageInfo = document.getElementById('upscaledImageInfo');
     const submitBtn = document.getElementById('submitBtn');
     const settingsBtn = document.getElementById('settingsBtn');
     const appContainer = document.querySelector('.app-container');
     const mainSidebar = document.querySelector('.sidebar');
     const mainPanel = document.querySelector('.main-panel');
     const sidebarMainContent = document.getElementById('sidebarMainContent');
     const sidebarSettingsNavContainer = document.getElementById('sidebarSettingsNav');

     const collapsibleHeaders = document.querySelectorAll('.collapsible-header');

     const initialView = document.getElementById('initialView');
     const imageDisplayView = document.getElementById('imageDisplayView');
     const comparisonView = document.getElementById('comparisonView');
     const errorDisplayArea = document.getElementById('errorDisplayArea');
     const dropAreaTarget = document.getElementById('dropAreaTarget');

     const displayedImage = document.getElementById('displayedImage');
     const originalComparisonImage = document.getElementById('originalComparisonImage');
     const processedComparisonImage = document.getElementById('processedComparisonImage');
     const comparisonSliderElement = document.querySelector('img-comparison-slider');

     const processingOverlay = document.getElementById('processingOverlay');
     const resultActionsOverlay = document.getElementById('resultActionsOverlay');
     const statusText = document.getElementById('statusText');
     const progressBar = document.getElementById('progressBar');
     const tileProgressText = document.getElementById('tileProgressText');

     const viewComparisonBtn = document.getElementById('viewComparisonBtn');
     const downloadResultLink = document.getElementById('downloadResultLink');
     const closeComparisonBtn = document.getElementById('closeComparisonBtn');
     const errorOkBtn = document.getElementById('errorOkBtn');
     const startOverBtnMain = document.getElementById('startOverBtnMain'); // Now Re-Process

     // Settings related elements from templates
     const settingsNavListTemplate = document.getElementById('settingsNavListToClone');
     const settingsTabsTemplate = document.getElementById('settingsTabsToClone');
     const settingsContentArea = document.getElementById('settingsContentArea'); // This is the main panel target
     const settingsContentTitle = document.getElementById('settingsContentTitle');


     const errorText = document.getElementById('errorText');

     let currentTaskId = null;
     let progressInterval = null;
     let originalImageObjectUrl = null;
     let originalImageName = "image.png";
     let originalImageDimensions = { w: 0, h: 0 };
     let currentProcessedImagePath = null;
     let currentLogBuffer = [];
     let isSettingsActive = false;
     let activeSettingsTabContent = null;


     fetchConfigOptions();
     setupEventListeners();
     switchToView('initialView');
     document.getElementById('currentYear').textContent = new Date().getFullYear();
     ['saveCropped', 'saveRestored', 'saveComparison'].forEach(id => {
          const checkbox = document.getElementById(id); if (checkbox) checkbox.checked = false;
     });

     function fetchConfigOptions() {
          fetch('/config_options')
               .then(response => response.json())
               .then(data => {
                    createCustomSelectButtons('faceEnhanceModelButtons', data.face_models, data.default_face_model);
                    createCustomSelectButtons('bgEnhanceModelButtons', data.bg_models, data.default_bg_model);
                    const upscaleOptions = [{ value: "2", name: "2x" }, { value: "4", name: "4x" }, { value: "8", name: "8x" }];
                    createCustomSelectButtons('overallUpscaleButtons', upscaleOptions, data.default_upscale.toString());
                    createCustomSelectButtons('deviceButtons', data.devices.map(d => ({ value: d, name: d.toUpperCase() })), 'auto');
                    createCustomSelectButtons('outputExtButtons', data.output_formats.map(f => ({ value: f, name: f.toUpperCase() })), 'auto');
                    const bgTileSizeEl = document.getElementById('bgTileSize');
                    if (bgTileSizeEl) bgTileSizeEl.value = data.default_bg_tile;
                    updateUpscaledImageInfo();
               })
               .catch(error => console.error('Error fetching config options:', error));
     }

     function createCustomSelectButtons(containerId, options, defaultValue) {
          const container = document.getElementById(containerId);
          const hiddenInputId = container.dataset.targetHidden;
          const hiddenInput = document.getElementById(hiddenInputId);
          if (!container || !hiddenInput) { console.error("Container or hidden input not found for", containerId); return; }
          container.innerHTML = '';
          options.forEach(option => {
               const button = document.createElement('button');
               button.type = 'button'; button.textContent = option.name; button.dataset.value = option.value;
               if (String(option.value) === String(defaultValue)) { button.classList.add('active'); hiddenInput.value = option.value; }
               button.addEventListener('click', function () {
                    container.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active'); hiddenInput.value = this.dataset.value;
                    if (containerId === 'overallUpscaleButtons') updateUpscaledImageInfo();
               });
               container.appendChild(button);
          });
     }

     function setupEventListeners() {
          if (dropAreaTarget) dropAreaTarget.addEventListener('click', () => inputFile.click());
          inputFile.addEventListener('change', handleImageSelection);
          ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
               if (initialView) initialView.addEventListener(eventName, preventDefaults, false);
               document.body.addEventListener(eventName, preventDefaults, false);
          });
          ['dragenter', 'dragover'].forEach(eventName => { if (initialView) initialView.addEventListener(eventName, () => initialView.classList.add('highlight'), false); });
          ['dragleave', 'drop'].forEach(eventName => { if (initialView) initialView.addEventListener(eventName, () => initialView.classList.remove('highlight'), false); });
          if (initialView) initialView.addEventListener('drop', handleDrop, false);

          submitBtn.addEventListener('click', handleSubmit);
          viewComparisonBtn.addEventListener('click', showComparisonView);
          closeComparisonBtn.addEventListener('click', hideComparisonView);
          if (startOverBtnMain) startOverBtnMain.addEventListener('click', handleReProcess);
          if (errorOkBtn) errorOkBtn.addEventListener('click', () => hideError(true));

          if (settingsBtn) settingsBtn.addEventListener('click', toggleSettingsMode);

          collapsibleHeaders.forEach(header => {
               header.addEventListener('click', function () {
                    this.classList.toggle('active'); const content = this.nextElementSibling;
                    if (content.style.maxHeight && content.style.maxHeight !== "0px") { content.style.maxHeight = "0px"; }
                    else { content.style.maxHeight = content.scrollHeight + "px"; }
               });
          });

          if (imageDisplayView) {
               imageDisplayView.addEventListener('mouseenter', () => {
                    if (currentProcessedImagePath && processingOverlay.style.display !== 'flex' && !isSettingsActive) {
                         imageDisplayView.classList.add('actions-visible');
                    }
               });
               imageDisplayView.addEventListener('mouseleave', () => {
                    imageDisplayView.classList.remove('actions-visible');
               });
          }
     }

     function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
     function handleDrop(e) { let dt = e.dataTransfer; let files = dt.files; if (files.length > 0) { inputFile.files = files; handleImageSelection({ target: inputFile }); } }

     function handleImageSelection(event) {
          const file = event.target.files[0];
          if (file) {
               if (isSettingsActive) toggleSettingsMode();
               originalImageName = file.name;
               if (originalImageObjectUrl) URL.revokeObjectURL(originalImageObjectUrl);
               originalImageObjectUrl = URL.createObjectURL(file);
               displayedImage.src = originalImageObjectUrl; originalComparisonImage.src = originalImageObjectUrl;
               const img = new Image();
               img.onload = () => {
                    originalImageDimensions.w = img.width; originalImageDimensions.h = img.height;
                    originalImageInfo.textContent = `Original: ${img.width}x${img.height}`;
                    updateUpscaledImageInfo();
               };
               img.src = originalImageObjectUrl;
               resetProcessingUIStates(); hideResultActionsOverlay();
               currentLogBuffer = [];
               const fullLogOutputEl = settingsContentArea.querySelector('#fullLogOutput');
               if (fullLogOutputEl) fullLogOutputEl.innerHTML = '';
               switchToView('imageDisplayView'); submitBtn.disabled = false;
               startOverBtnMain.disabled = false;
          }
     }

     function updateUpscaledImageInfo() {
          const overallUpscaleHidden = document.getElementById('overallUpscaleHidden');
          if (originalImageDimensions.w > 0 && originalImageDimensions.h > 0 && overallUpscaleHidden && overallUpscaleHidden.value) {
               const scale = parseInt(overallUpscaleHidden.value, 10);
               const newW = originalImageDimensions.w * scale; const newH = originalImageDimensions.h * scale;
               upscaledImageInfo.textContent = `ScaleX to ${newW}x${newH}`;
          } else { upscaledImageInfo.textContent = 'ScaleX to ?x?'; }
     }

     function formatStatusText(statusString) {
          if (!statusString) return 'Processing...';
          return statusString.replace(/_/g, ' ').toLowerCase().split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
     }

     function handleSubmit() {
          if (!inputFile.files || inputFile.files.length === 0) {
               if (!originalImageObjectUrl) {
                    alert('Please select an image.'); return;
               }
          }
          if (isSettingsActive) toggleSettingsMode();
          switchToView('imageDisplayView'); showProcessingOverlay(true); hideResultActionsOverlay();
          submitBtn.disabled = true; startOverBtnMain.disabled = true;
          statusText.textContent = 'Uploading & Preparing...';
          progressBar.style.width = '0%'; progressBar.textContent = '0%';
          tileProgressText.style.display = 'none';
          currentLogBuffer = [];
          const fullLogOutputEl = settingsContentArea.querySelector('#fullLogOutput');
          if (fullLogOutputEl) fullLogOutputEl.innerHTML = '';

          const formData = new FormData(scalexForm);
          if (inputFile.files[0]) {
               formData.append('inputFile', inputFile.files[0]);
          } else if (originalImageObjectUrl && originalImageName) {
               // For re-processing, the file might not be in inputFile.files.
               // This relies on the backend being able to re-fetch/re-use the uploaded file if only form data is sent.
               // Or, a more complex solution to re-construct File object from blob URL (not done here).
               // The 'handleReProcess' function tries to ensure inputFile.files[0] is set if possible.
               console.warn("Attempting to process without a direct file input, using stored original image name.");
          }


          fetch('/process', { method: 'POST', body: formData })
               .then(response => response.json())
               .then(data => {
                    if (data.error) { showError(data.error); resetToImageSelectedState(); return; }
                    currentTaskId = data.task_id;
                    appendLog(`Task ${currentTaskId} started for ${originalImageName}.`);
                    statusText.textContent = data.current_step_description || formatStatusText(data.status) || `Processing (Task ID: ${currentTaskId})`;
                    if (data.original_uploaded_path) { originalComparisonImage.src = `/uploads/${data.original_uploaded_path}?t=${new Date().getTime()}`; }
                    else if (originalImageObjectUrl) { originalComparisonImage.src = originalImageObjectUrl; }
                    startProgressCheck();
               })
               .catch(error => { console.error('Error submitting:', error); showError('Failed to start: ' + error.message); resetToImageSelectedState(); });
     }

     function handleReProcess() {
          if (!originalImageObjectUrl) {
               alert("No image loaded to re-process.");
               return;
          }
          if (inputFile.files.length === 0) {
               // If the file input is empty, try to re-populate it (won't work for security reasons if it was cleared)
               // Best approach is to just submit current form data if backend supports re-using original upload.
               // Or prompt user to re-select if strict file upload is needed by backend.
               console.log("Re-processing with current settings, assuming backend can use original upload.");
          }
          handleSubmit();
     }


     function startProgressCheck() {
          if (progressInterval) clearInterval(progressInterval);
          progressInterval = setInterval(() => {
               if (!currentTaskId) { clearInterval(progressInterval); return; }
               fetch(`/progress/${currentTaskId}`)
                    .then(response => response.json())
                    .then(data => {
                         if (data.error) { showError(`Progress error: ${data.error}`); clearInterval(progressInterval); resetToImageSelectedState(); return; }

                         if (data.logs && data.logs.length > 0) { data.logs.forEach(logMsg => appendLog(logMsg)); }

                         statusText.textContent = data.current_step_description || formatStatusText(data.status) || 'Processing...';
                         const progress = data.progress || 0;
                         progressBar.style.width = `${progress}%`; progressBar.textContent = `${Math.round(progress)}%`;

                         if (data.tile_progress && data.tile_progress.total > 0) {
                              tileProgressText.textContent = `BG Tiling: ${data.tile_progress.current}/${data.tile_progress.total} (${Math.round(data.tile_progress.percentage)}%)`;
                              tileProgressText.style.display = 'block';
                         } else { tileProgressText.style.display = 'none'; }

                         if (data.status === 'completed') {
                              clearInterval(progressInterval); showProcessingOverlay(false);
                              if (data.result_path) {
                                   currentProcessedImagePath = `/outputs/${data.result_path}`;
                                   displayedImage.src = `${currentProcessedImagePath}?t=${new Date().getTime()}`;
                                   processedComparisonImage.src = `${currentProcessedImagePath}?t=${new Date().getTime()}`;

                                   let baseName = originalImageName;
                                   const nameParts = originalImageName.split('.');
                                   if (nameParts.length > 1) baseName = nameParts.slice(0, -1).join('.');

                                   let extension = "png";
                                   const resultPathParts = data.result_path.split(/[\\/]/).pop().split('.');
                                   if (resultPathParts.length > 1) extension = resultPathParts.pop();

                                   downloadResultLink.href = currentProcessedImagePath;
                                   downloadResultLink.download = `Enhanced_${baseName}.${extension}`;
                              } else {
                                   appendLog("[INFO] Main output not available for display.");
                                   showError("Processing finished, but no main output image was found for display.");
                                   hideResultActionsOverlay();
                                   displayedImage.src = originalImageObjectUrl || "#";
                              }
                              submitBtn.disabled = false; startOverBtnMain.disabled = false;
                              switchToView('imageDisplayView');
                         } else if (data.status === 'error') {
                              clearInterval(progressInterval);
                              showError(`Processing error: ${data.error || 'Unknown error occurred.'}`);
                              resetToImageSelectedState();
                         } else if (!data.thread_active && data.status !== 'completed' && data.status !== 'error') {
                              clearInterval(progressInterval);
                              if (!data.error) { showError('Processing stopped unexpectedly.'); }
                              resetToImageSelectedState();
                         }
                    })
                    .catch(error => {
                         console.error('Error fetching progress:', error);
                         showError('Connection error while fetching progress.');
                         clearInterval(progressInterval);
                         resetToImageSelectedState();
                    });
          }, 1000);
     }

     function switchToView(viewId, forceExitSettings = false) {
          if (isSettingsActive && !forceExitSettings && !['settingsContentArea', 'errorDisplayArea'].includes(viewId)) {
               if (viewId === 'errorDisplayArea') {
                    const errorEl = document.getElementById(viewId);
                    if (errorEl) errorEl.classList.add('active');
               }
               return;
          }
          if (forceExitSettings && isSettingsActive) {
               toggleSettingsMode();
          }

          ['initialView', 'imageDisplayView', 'comparisonView', 'errorDisplayArea', 'settingsContentArea'].forEach(id => {
               const el = document.getElementById(id);
               if (el) el.classList.remove('active');
          });

          const targetView = document.getElementById(viewId);
          if (targetView) {
               targetView.classList.add('active');
          }

          if (viewId !== 'imageDisplayView') {
               showProcessingOverlay(false);
               hideResultActionsOverlay();
          }
          if (viewId === 'comparisonView') {
               if (comparisonView) comparisonView.style.display = 'flex';
          } else {
               if (comparisonView) comparisonView.style.display = 'none';
          }
     }


     function showProcessingOverlay(show) { if (processingOverlay) processingOverlay.style.display = show ? 'flex' : 'none'; }
     function hideResultActionsOverlay() { if (imageDisplayView) imageDisplayView.classList.remove('actions-visible'); }

     function showComparisonView() {
          if (!originalComparisonImage.src || originalComparisonImage.src.endsWith('#') || originalComparisonImage.src === window.location.href ||
               !processedComparisonImage.src || processedComparisonImage.src.endsWith('#') || processedComparisonImage.src === window.location.href) {
               showError("Cannot compare: Original or processed image is missing or invalid.");
               return;
          }
          if (comparisonSliderElement) comparisonSliderElement.reset();
          hideResultActionsOverlay();
          comparisonView.classList.add('active');
          comparisonView.style.display = 'flex';
     }
     function hideComparisonView() {
          comparisonView.classList.remove('active');
          comparisonView.style.display = 'none';
          switchToView('imageDisplayView');
     }

     function resetProcessingUIStates() {
          if (progressInterval) clearInterval(progressInterval); currentTaskId = null;
          showProcessingOverlay(false); hideResultActionsOverlay();
          statusText.textContent = 'Status: Ready.'; progressBar.style.width = '0%'; progressBar.textContent = '0%';
          tileProgressText.style.display = 'none';
          hideError(false);
          submitBtn.disabled = true; startOverBtnMain.disabled = true;
     }

     function resetToImageSelectedState() {
          showProcessingOverlay(false); hideResultActionsOverlay();
          switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView', true);
          submitBtn.disabled = !originalImageObjectUrl;
          startOverBtnMain.disabled = !originalImageObjectUrl;
          if (progressInterval) clearInterval(progressInterval);
     }

     function resetToInitialState() { // Full UI reset
          if (isSettingsActive) toggleSettingsMode();
          if (originalImageObjectUrl) { URL.revokeObjectURL(originalImageObjectUrl); originalImageObjectUrl = null; }
          currentProcessedImagePath = null; originalImageName = "image.png"; inputFile.value = '';
          displayedImage.src = '#'; originalComparisonImage.src = '#'; processedComparisonImage.src = '#';
          if (comparisonSliderElement) comparisonSliderElement.reset();
          originalImageDimensions = { w: 0, h: 0 }; originalImageInfo.textContent = 'Original: ?x?';
          updateUpscaledImageInfo();
          resetProcessingUIStates();
          switchToView('initialView');
          fetchConfigOptions();
          collapsibleHeaders.forEach(header => {
               if (header.classList.contains('active')) {
                    header.classList.remove('active');
                    const content = header.nextElementSibling;
                    content.style.maxHeight = "0px";
               }
          });
          ['saveCropped', 'saveRestored', 'saveComparison'].forEach(id => { const cb = document.getElementById(id); if (cb) cb.checked = false; });
          currentLogBuffer = [];
          const fullLogOutputEl = settingsContentArea.querySelector('#fullLogOutput');
          if (fullLogOutputEl) fullLogOutputEl.innerHTML = '';
          submitBtn.disabled = true; startOverBtnMain.disabled = true;
     }

     function appendLogToElement(message, element, isNew) {
          if (!element) return;
          const logEntry = document.createElement('div');
          logEntry.classList.add('log-entry');
          let level = 'info';
          const lowerMessage = message.toLowerCase();
          if (lowerMessage.includes('[error]') || lowerMessage.includes('error:')) { level = 'error'; }
          else if (lowerMessage.includes('[warning]') || lowerMessage.includes('warning:')) { level = 'warning'; }
          else if (lowerMessage.includes('[debug]')) { level = 'debug'; }
          logEntry.classList.add(`log-level-${level}`);
          logEntry.textContent = message;
          element.appendChild(logEntry);
          if (isNew || element === settingsContentArea.querySelector('#fullLogOutput')) {
               element.scrollTop = element.scrollHeight;
          }
     }
     function appendLog(message) {
          currentLogBuffer.push(message);
          if (currentLogBuffer.length > 300) { currentLogBuffer.shift(); }
          const fullLogOutputEl = activeSettingsTabContent?.querySelector('#fullLogOutput'); // Use active tab's log
          if (fullLogOutputEl && isSettingsActive && activeSettingsTabContent?.id === 'settingsLogs') {
               appendLogToElement(message, fullLogOutputEl, true);
          }
     }

     function showError(message) {
          errorText.textContent = message;
          showProcessingOverlay(false);
          hideResultActionsOverlay();
          switchToView('errorDisplayArea');
          console.error("UI Error:", message);
          appendLog(`[ERROR] UI: ${message}`);
     }
     function hideError(forceExitSettingsOnErrorHide = false) {
          if (isSettingsActive && !forceExitSettingsOnErrorHide) {
               const errorEl = document.getElementById('errorDisplayArea');
               if (errorEl) errorEl.classList.remove('active');
          } else {
               switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView', true);
          }
     }

     function toggleSettingsMode() {
          isSettingsActive = !isSettingsActive;
          appContainer.classList.toggle('settings-mode-active');

          if (isSettingsActive) {
               sidebarMainContent.style.display = 'none';
               sidebarSettingsNavContainer.innerHTML = '';
               const navListClone = settingsNavListTemplate.querySelector('ul').cloneNode(true);
               sidebarSettingsNavContainer.appendChild(navListClone);
               sidebarSettingsNavContainer.style.display = 'block';

               sidebarSettingsNavContainer.querySelectorAll('.settings-nav-btn').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                         sidebarSettingsNavContainer.querySelectorAll('.settings-nav-btn').forEach(b => b.classList.remove('active'));
                         e.currentTarget.classList.add('active');
                         const targetTabId = e.currentTarget.dataset.targetTab;
                         displaySettingsTab(targetTabId);
                    });
               });

               ['initialView', 'imageDisplayView', 'comparisonView', 'errorDisplayArea'].forEach(id => {
                    const el = document.getElementById(id); if (el) el.classList.remove('active');
               });
               hideResultActionsOverlay(); showProcessingOverlay(false);

               settingsContentArea.classList.add('active'); // This is now a .view-panel
               settingsContentArea.style.display = 'flex'; // Ensure it is flex
               settingsContentTitle.style.display = 'flex';


               const currentActiveNav = sidebarSettingsNavContainer.querySelector('.settings-nav-btn.active');
               if (currentActiveNav) { displaySettingsTab(currentActiveNav.dataset.targetTab); }
               else if (sidebarSettingsNavContainer.querySelector('.settings-nav-btn')) {
                    sidebarSettingsNavContainer.querySelector('.settings-nav-btn').click();
               }
               settingsBtn.innerHTML = '<i class="fas fa-arrow-left"></i>'; settingsBtn.title = "Back to Main View";
          } else {
               sidebarMainContent.style.display = 'flex';
               sidebarSettingsNavContainer.style.display = 'none';
               settingsContentArea.classList.remove('active');
               settingsContentArea.style.display = 'none';
               settingsContentTitle.style.display = 'none';

               switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView');
               settingsBtn.innerHTML = '<i class="fas fa-cog"></i>'; settingsBtn.title = "Settings";
          }
     }

     function displaySettingsTab(tabId) {
          if (!settingsContentArea) return;
          if (activeSettingsTabContent && activeSettingsTabContent.parentNode === settingsContentArea) {
               settingsContentArea.removeChild(activeSettingsTabContent);
          }
          activeSettingsTabContent = null;
          const tabContentClone = settingsTabsTemplate.querySelector(`#${tabId}`)?.cloneNode(true);

          if (tabContentClone) {
               settingsContentArea.appendChild(tabContentClone);
               tabContentClone.classList.add('active');
               activeSettingsTabContent = tabContentClone;

               if (tabId === 'settingsOutput') {
                    const btnSave = activeSettingsTabContent.querySelector('#saveOutputDirectoryBtn');
                    if (btnSave) btnSave.addEventListener('click', handleSaveOutputDirectory);
                    loadCurrentOutputDirectory();
               }
               if (tabId === 'settingsActions') {
                    const btnRestart = activeSettingsTabContent.querySelector('#restartBackendBtn');
                    const btnClear = activeSettingsTabContent.querySelector('#clearBackendDirsBtn');
                    if (btnRestart) btnRestart.addEventListener('click', handleRestartBackend);
                    if (btnClear) btnClear.addEventListener('click', handleClearBackendDirs);
               }
               if (tabId === 'settingsSystemInfo') { fetchSystemInfo(); }
               if (tabId === 'settingsLogs') {
                    const fullLogEl = activeSettingsTabContent.querySelector('#fullLogOutput');
                    if (fullLogEl) {
                         fullLogEl.innerHTML = '';
                         currentLogBuffer.forEach(logMsg => appendLogToElement(logMsg, fullLogEl, false));
                         fullLogEl.scrollTop = fullLogEl.scrollHeight;
                    }
               }
          }
     }

     function fetchSystemInfo() {
          const sysInfoContentEl = activeSettingsTabContent?.querySelector('#systemInfoContent');
          if (!sysInfoContentEl) return;
          sysInfoContentEl.innerHTML = '<p><i class="fas fa-spinner fa-spin"></i> Loading system info...</p>';
          fetch('/system_info').then(r => r.json()).then(data => {
               if (data.error) { sysInfoContentEl.innerHTML = `<p>Error: ${data.error}</p>`; return; }
               let html = `<p><strong>App Version:</strong> ${data.app_version}</p><p><strong>Python:</strong> ${data.python_version}</p><p><strong>PyTorch:</strong> ${data.torch_version}</p><p><strong>CUDA:</strong> ${data.cuda_available ? 'Yes' : 'No'}</p>`;
               if (data.gpus && data.gpus.length > 0) html += `<p><strong>GPUs:</strong> ${data.gpus.join(', ')}</p>`;
               html += `<p><strong>OS:</strong> ${data.os}</p><p><strong>CPU:</strong> ${data.cpu}</p><p><strong>RAM:</strong> ${data.ram}</p>`;
               sysInfoContentEl.innerHTML = html;

               const outputFolderDispEl = activeSettingsTabContent?.querySelector('#defaultOutputFolderDisplay');
               const customOutputDirInputEl = activeSettingsTabContent?.querySelector('#customOutputDirectory');
               if (outputFolderDispEl && data.default_output_folder) outputFolderDispEl.textContent = data.default_output_folder;
               if (customOutputDirInputEl && data.default_output_folder && !customOutputDirInputEl.value) customOutputDirInputEl.value = data.default_output_folder;
          }).catch(err => { sysInfoContentEl.innerHTML = `<p>Failed to fetch: ${err}</p>`; });
     }

     function handleRestartBackend() {
          const restartStatusEl = activeSettingsTabContent?.querySelector('#restartStatus');
          if (!confirm("Are you sure you want to attempt to restart the backend?")) return;
          if (restartStatusEl) restartStatusEl.textContent = "Attempting restart...";
          fetch('/restart_backend', { method: 'POST' }).then(r => r.json()).then(data => {
               if (restartStatusEl) restartStatusEl.textContent = data.message || "Restart command sent.";
               alert((data.message || "Restart command sent.") + "\nA manual refresh might be needed.");
          }).catch(err => { if (restartStatusEl) restartStatusEl.textContent = `Error: ${err}`; alert("Failed to send restart command."); });
     }

     function loadCurrentOutputDirectory() {
          const customOutputDirInputEl = activeSettingsTabContent?.querySelector('#customOutputDirectory');
          const outputFolderDispEl = activeSettingsTabContent?.querySelector('#defaultOutputFolderDisplay');
          if (!customOutputDirInputEl && !outputFolderDispEl) return;

          fetch('/get_output_directory').then(r => r.json()).then(data => {
               if (data.output_directory) {
                    if (customOutputDirInputEl) customOutputDirInputEl.value = data.output_directory;
                    if (outputFolderDispEl) outputFolderDispEl.textContent = data.output_directory;
               }
          }).catch(err => console.error('Error fetching output dir:', err));
     }

     function handleSaveOutputDirectory() {
          const customOutputDirInputEl = activeSettingsTabContent?.querySelector('#customOutputDirectory');
          const outputDirStatusEl = activeSettingsTabContent?.querySelector('#outputDirStatus');
          if (!customOutputDirInputEl || !outputDirStatusEl) return;
          const newPath = customOutputDirInputEl.value.trim();
          if (!newPath) { outputDirStatusEl.textContent = 'Path cannot be empty.'; return; }
          outputDirStatusEl.textContent = 'Saving...';
          fetch('/set_output_directory', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ output_directory: newPath }) })
               .then(r => r.json()).then(data => {
                    if (data.error) { outputDirStatusEl.textContent = `Error: ${data.error}`; appendLog(`[ERROR] Set output dir: ${data.error}`); }
                    else {
                         outputDirStatusEl.textContent = data.message || 'Path saved!';
                         const outputFolderDispEl = activeSettingsTabContent?.querySelector('#defaultOutputFolderDisplay');
                         if (outputFolderDispEl && data.new_path) outputFolderDispEl.textContent = data.new_path;
                         appendLog(`[INFO] Output directory set to: ${data.new_path}`);
                         if (activeSettingsTabContent?.querySelector('#systemInfoContent')) { fetchSystemInfo(); }
                    }
               }).catch(err => { outputDirStatusEl.textContent = 'Error saving.'; appendLog(`[ERROR] Set output dir: ${err}`); });
     }

     function handleClearBackendDirs() {
          const clearDirsStatusEl = activeSettingsTabContent?.querySelector('#clearDirsStatus');
          if (!confirm("Clear ALL images from input/output dirs? IRREVERSIBLE.")) return;
          if (clearDirsStatusEl) clearDirsStatusEl.textContent = 'Clearing...';
          fetch('/clear_backend_dirs', { method: 'POST' }).then(r => r.json()).then(data => {
               if (clearDirsStatusEl) clearDirsStatusEl.textContent = data.message || 'Clear cmd sent.';
               appendLog(`[INFO] Clear backend: ${data.message}`);
               if (data.errors && data.errors.length > 0) data.errors.forEach(e => appendLog(`[ERROR] Clear backend: ${e}`));
          }).catch(err => { if (clearDirsStatusEl) clearDirsStatusEl.textContent = 'Error sending cmd.'; appendLog(`[ERROR] Clear backend: ${err}`); });
     }
     loadCurrentOutputDirectory();
});