document.addEventListener('DOMContentLoaded', function () {
     // --- DOM Element Selectors ---
     const scalexForm = document.getElementById('scalexForm');
     const inputFile = document.getElementById('inputFile');
     const originalImageInfo = document.getElementById('originalImageInfo');
     const upscaledImageInfo = document.getElementById('upscaledImageInfo');
     const submitBtn = document.getElementById('submitBtn');
     const settingsBtn = document.getElementById('settingsBtn');
     const appContainer = document.querySelector('.app-container');
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
     const startOverBtnMain = document.getElementById('startOverBtnMain');

     const settingsNavListTemplate = document.getElementById('settingsNavListToClone');
     const settingsTabsTemplate = document.getElementById('settingsTabsToClone');
     const settingsContentArea = document.getElementById('settingsContentArea');
     const settingsContentTitle = document.getElementById('settingsContentTitle');

     const errorText = document.getElementById('errorText');

     let currentTaskId = null;
     let progressInterval = null;
     let originalImageObjectUrl = null;
     let originalImageName = "image.png";
     let originalImageDimensions = { w: 0, h: 0 };
     let currentProcessedImageRelativePath = null;
     let currentFullProcessedImageUrl = null;
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
          // Mocking backend response for local testing if /config_options is not available
          const mockConfig = {
               face_models: [{ value: "GFPGANv1.4", name: "GFPGANv1.4" }, { value: "CodeFormer", name: "CodeFormer" }],
               default_face_model: "GFPGANv1.4",
               bg_models: [{ value: "RealESRGAN_x4plus", name: "RealESRGAN x4+" }, { value: "RealESRGAN_x4plus_anime_6B", name: "ESRGAN Anime" }],
               default_bg_model: "RealESRGAN_x4plus",
               default_upscale: 2,
               devices: ["auto", "cpu", "cuda"],
               output_formats: ["auto", "png", "jpg", "webp"],
               default_bg_tile: 400
          };

          const useMock = false; // Set to true to use mock data if backend isn't running

          if (useMock) {
               Promise.resolve(mockConfig).then(data => {
                    createCustomSelectButtons('faceEnhanceModelButtons', data.face_models, data.default_face_model);
                    createCustomSelectButtons('bgEnhanceModelButtons', data.bg_models, data.default_bg_model);
                    const upscaleOptions = [{ value: "2", name: "2x" }, { value: "4", name: "4x" }, { value: "8", name: "8x" }];
                    createCustomSelectButtons('overallUpscaleButtons', upscaleOptions, data.default_upscale.toString());
                    createCustomSelectButtons('deviceButtons', data.devices.map(d => ({ value: d, name: d.toUpperCase() })), 'auto');
                    createCustomSelectButtons('outputExtButtons', data.output_formats.map(f => ({ value: f, name: f.toUpperCase() })), 'auto');
                    const bgTileSizeEl = document.getElementById('bgTileSize');
                    if (bgTileSizeEl) bgTileSizeEl.value = data.default_bg_tile;
                    updateUpscaledImageInfo();
               });
               return;
          }


          fetch('/config_options')
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
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
          if (!Array.isArray(options)) {
               console.error(`Options for ${containerId} is not an array:`, options);
               options = []; // Prevent further errors
          }
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

          if (displayedImage && imageDisplayView && resultActionsOverlay) {
               displayedImage.addEventListener('mouseenter', () => {
                    if (currentFullProcessedImageUrl && processingOverlay.style.display !== 'flex' && !isSettingsActive) {
                         imageDisplayView.classList.add('actions-visible');
                    }
               });

               imageDisplayView.addEventListener('mouseleave', () => {
                    imageDisplayView.classList.remove('actions-visible');
               });
          } else {
               console.error("Error setting up hover listeners: One or more elements (displayedImage, imageDisplayView, resultActionsOverlay) not found.");
          }
     }

     function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
     function handleDrop(e) { let dt = e.dataTransfer; let files = dt.files; if (files.length > 0) { inputFile.files = files; handleImageSelection({ target: inputFile }); } }

     function clearProcessedImageData() {
          console.log("Clearing processed image data.");
          if (processedComparisonImage) processedComparisonImage.src = '#';
          currentProcessedImageRelativePath = null;
          currentFullProcessedImageUrl = null;
          if (downloadResultLink) {
               downloadResultLink.href = "#";
               downloadResultLink.removeAttribute('download');
          }
     }

     function handleImageSelection(event) {
          const file = event.target.files[0];
          if (file) {
               if (isSettingsActive) toggleSettingsMode();
               originalImageName = file.name;
               if (originalImageObjectUrl) URL.revokeObjectURL(originalImageObjectUrl);
               originalImageObjectUrl = URL.createObjectURL(file);

               console.log("New image selected. Original Object URL:", originalImageObjectUrl);
               displayedImage.src = originalImageObjectUrl;
               originalComparisonImage.src = originalImageObjectUrl;

               clearProcessedImageData();

               const img = new Image();
               img.onload = () => {
                    originalImageDimensions.w = img.width; originalImageDimensions.h = img.height;
                    originalImageInfo.textContent = `Original: ${img.width}x${img.height}`;
                    updateUpscaledImageInfo();
               };
               img.src = originalImageObjectUrl;

               resetProcessingUIStates();
               hideResultActionsOverlay();
               currentLogBuffer = [];
               const fullLogOutputEl = settingsContentArea.querySelector('#fullLogOutput');
               if (fullLogOutputEl) fullLogOutputEl.innerHTML = '';

               switchToView('imageDisplayView');
               submitBtn.disabled = false;
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

          clearProcessedImageData();

          switchToView('imageDisplayView');
          showProcessingOverlay(true);
          hideResultActionsOverlay();
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
               console.warn("Attempting to process using stored original image name, as inputFile is empty.");
          }

          fetch('/process', { method: 'POST', body: formData })
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (data.error) { showError(data.error); resetToImageSelectedState(); return; }
                    currentTaskId = data.task_id;
                    appendLog(`Task ${currentTaskId} started for ${originalImageName}.`);
                    statusText.textContent = data.current_step_description || formatStatusText(data.status) || `Processing (Task ID: ${currentTaskId})`;

                    if (data.original_uploaded_path) {
                         originalComparisonImage.src = `/uploads/${data.original_uploaded_path.replace(/\\/g, '/')}?t=${new Date().getTime()}`;
                         console.log("Original comparison image set from server path:", originalComparisonImage.src);
                    } else if (originalImageObjectUrl) {
                         originalComparisonImage.src = originalImageObjectUrl;
                         console.log("Original comparison image set from blob URL:", originalComparisonImage.src);
                    }
                    startProgressCheck();
               })
               .catch(error => { console.error('Error submitting:', error); showError('Failed to start: ' + error.message); resetToImageSelectedState(); });
     }

     function handleReProcess() {
          if (!originalImageObjectUrl) {
               alert("No image loaded to re-process.");
               return;
          }
          handleSubmit();
     }

     function startProgressCheck() {
          if (progressInterval) clearInterval(progressInterval);
          progressInterval = setInterval(() => {
               if (!currentTaskId) { clearInterval(progressInterval); return; }
               fetch(`/progress/${currentTaskId}`)
                    .then(response => {
                         if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                         return response.json();
                    })
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
                              clearInterval(progressInterval);
                              showProcessingOverlay(false);
                              if (data.result_path) {
                                   currentProcessedImageRelativePath = data.result_path.replace(/\\/g, '/');
                                   currentFullProcessedImageUrl = `/outputs/${currentProcessedImageRelativePath}?t=${new Date().getTime()}`;
                                   console.log("Processing complete. Processed image URL:", currentFullProcessedImageUrl);

                                   displayedImage.src = currentFullProcessedImageUrl;
                                   processedComparisonImage.src = currentFullProcessedImageUrl;

                                   let baseName = originalImageName.substring(0, originalImageName.lastIndexOf('.')) || originalImageName;
                                   let extension = (currentProcessedImageRelativePath.split('.').pop() || 'png').toLowerCase();

                                   downloadResultLink.href = currentFullProcessedImageUrl;
                                   downloadResultLink.download = `Enhanced_${baseName}.${extension}`;
                              } else {
                                   appendLog("[INFO] Main output not available for display.");
                                   showError("Processing finished, but no main output image was found for display.");
                                   hideResultActionsOverlay();
                                   displayedImage.src = originalImageObjectUrl || "#";
                                   clearProcessedImageData();
                              }
                              submitBtn.disabled = false; startOverBtnMain.disabled = false;
                              switchToView('imageDisplayView');
                         } else if (data.status === 'error') {
                              clearInterval(progressInterval);
                              showError(`Processing error: ${data.error_message || data.error || 'Unknown error occurred.'}`);
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
          console.log(`Switching to view: ${viewId}, forceExitSettings: ${forceExitSettings}, isSettingsActive: ${isSettingsActive}`);

          if (isSettingsActive && !forceExitSettings && !['settingsContentArea', 'errorDisplayArea'].includes(viewId)) {
               if (viewId === 'errorDisplayArea') {
                    const errorEl = document.getElementById(viewId);
                    if (errorEl) errorEl.classList.add('active');
                    console.log(`${viewId} activated (over settings).`);
               } else {
                    console.log(`Attempted to switch to ${viewId} while settings active and not forced/allowed. No switch.`);
                    return;
               }
          } else {
               if (forceExitSettings && isSettingsActive) {
                    toggleSettingsMode();
                    return;
               }

               ['initialView', 'imageDisplayView', 'comparisonView', 'errorDisplayArea', 'settingsContentArea'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                         el.classList.remove('active');
                    }
               });

               const targetView = document.getElementById(viewId);
               if (targetView) {
                    targetView.classList.add('active');
                    console.log(`${viewId} activated. Class list:`, targetView.classList.toString());
               } else {
                    console.error(`Target view "${viewId}" not found!`);
               }
          }

          // Hide overlays unless explicitly on imageDisplayView (and not in settings)
          // or comparisonView (where overlays are not relevant).
          if (viewId !== 'imageDisplayView' || isSettingsActive) {
               if (viewId !== 'comparisonView') { // Keep resultActionsOverlay potentially for comparison view if logic changes
                    showProcessingOverlay(false);
                    hideResultActionsOverlay();
               } else {
                    showProcessingOverlay(false); // Definitely hide processing for comparison
               }
          }
          // Ensure processing overlay is off for these specific views
          if (viewId === 'initialView' || viewId === 'comparisonView' || (isSettingsActive && viewId === 'settingsContentArea')) {
               showProcessingOverlay(false);
          }
     }


     function showProcessingOverlay(show) {
          if (processingOverlay) {
               const newDisplay = show ? 'flex' : 'none';
               if (processingOverlay.style.display !== newDisplay) {
                    processingOverlay.style.display = newDisplay;
                    console.log(`Processing overlay display set to: ${newDisplay}`);
               }
          }
     }
     function hideResultActionsOverlay() {
          if (imageDisplayView && imageDisplayView.classList.contains('actions-visible')) {
               imageDisplayView.classList.remove('actions-visible');
               console.log("Result actions overlay hidden (actions-visible class removed).");
          }
     }

     function showComparisonView() {
          console.log("Attempting to show comparison view.");
          console.log("Original comparison src:", originalComparisonImage.src);
          console.log("Processed comparison src:", processedComparisonImage.src);

          const isValidHttpUrl = (string) => {
               if (!string) return false;
               try { new URL(string, window.location.origin); } catch (_) { return false; }
               return string.startsWith('http:') || string.startsWith('https:') || string.startsWith('blob:') || string.startsWith('/');
          }
          const isPlaceholderSrc = (src) => !src || src.endsWith('#') || src === window.location.href || src === (new URL('#', window.location.href)).toString();

          const originalSrcValid = isValidHttpUrl(originalComparisonImage.src) && !isPlaceholderSrc(originalComparisonImage.src);
          const processedSrcValid = isValidHttpUrl(processedComparisonImage.src) && !isPlaceholderSrc(processedComparisonImage.src);

          if (!originalSrcValid || !processedSrcValid) {
               let missing = [];
               if (!originalSrcValid) missing.push("original");
               if (!processedSrcValid) missing.push("processed");
               const errorMessage = `Cannot compare: The ${missing.join(' and ')} image source is missing or invalid.
Original: ${originalComparisonImage.src} (Valid: ${originalSrcValid})
Processed: ${processedComparisonImage.src} (Valid: ${processedSrcValid})`;
               showError(errorMessage);
               console.error(errorMessage);
               return;
          }

          if (comparisonSliderElement) {
               comparisonSliderElement.setAttribute('value', '0.5');
               console.log("Comparison slider 'value' attribute set to 0.5.");
          } else {
               console.warn("comparisonSliderElement not found when trying to show comparison view.");
          }
          hideResultActionsOverlay(); // Hide actions from main image view before switching
          switchToView('comparisonView');
     }

     function hideComparisonView() {
          console.log("Hiding comparison view.");
          switchToView('imageDisplayView');
     }

     function resetProcessingUIStates() {
          if (progressInterval) clearInterval(progressInterval); currentTaskId = null;
          showProcessingOverlay(false);
          hideResultActionsOverlay();
          statusText.textContent = 'Status: Ready.'; progressBar.style.width = '0%'; progressBar.textContent = '0%';
          tileProgressText.style.display = 'none';
          submitBtn.disabled = true; startOverBtnMain.disabled = true;
          console.log("Processing UI states reset.");
     }

     function resetToImageSelectedState() {
          showProcessingOverlay(false);
          hideResultActionsOverlay();
          clearProcessedImageData();

          switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView', true);
          submitBtn.disabled = !originalImageObjectUrl;
          startOverBtnMain.disabled = !originalImageObjectUrl;
          if (progressInterval) clearInterval(progressInterval);
          console.log("Reset to image selected state.");
     }

     function resetToInitialState() {
          console.log("Resetting to initial state.");
          if (isSettingsActive) toggleSettingsMode();
          if (originalImageObjectUrl) { URL.revokeObjectURL(originalImageObjectUrl); originalImageObjectUrl = null; }

          clearProcessedImageData();
          originalImageName = "image.png";
          if (inputFile) inputFile.value = '';

          if (displayedImage) displayedImage.src = '#';
          if (originalComparisonImage) originalComparisonImage.src = '#';

          originalImageDimensions = { w: 0, h: 0 };
          if (originalImageInfo) originalImageInfo.textContent = 'Original: ?x?';
          updateUpscaledImageInfo();

          resetProcessingUIStates();
          switchToView('initialView');
          fetchConfigOptions();
          collapsibleHeaders.forEach(header => {
               if (header.classList.contains('active')) {
                    header.classList.remove('active');
                    const content = header.nextElementSibling;
                    if (content) content.style.maxHeight = "0px";
               }
          });
          ['saveCropped', 'saveRestored', 'saveComparison'].forEach(id => {
               const cb = document.getElementById(id); if (cb) cb.checked = false;
          });
          currentLogBuffer = [];
          const fullLogOutputEl = settingsContentArea.querySelector('#fullLogOutput');
          if (fullLogOutputEl) fullLogOutputEl.innerHTML = '';
          if (submitBtn) submitBtn.disabled = true;
          if (startOverBtnMain) startOverBtnMain.disabled = true;
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
          if (isNew || (element === settingsContentArea.querySelector('#fullLogOutput') && element.scrollHeight > element.clientHeight)) {
               element.scrollTop = element.scrollHeight;
          }
     }
     function appendLog(message) {
          currentLogBuffer.push(message);
          if (currentLogBuffer.length > 300) {
               currentLogBuffer.shift();
          }
          const fullLogOutputEl = activeSettingsTabContent?.querySelector('#fullLogOutput');
          if (fullLogOutputEl && isSettingsActive && activeSettingsTabContent?.id === 'settingsLogs') {
               appendLogToElement(message, fullLogOutputEl, true);
          }
     }

     function showError(message) {
          console.error("UI Error Displayed:", message);
          errorText.textContent = message;
          showProcessingOverlay(false);
          hideResultActionsOverlay();
          switchToView('errorDisplayArea');
     }

     function hideError(forceExitSettingsOnErrorHide = false) {
          console.log(`Hiding error. forceExitSettings: ${forceExitSettingsOnErrorHide}, isSettingsActive: ${isSettingsActive}`);
          const errorEl = document.getElementById('errorDisplayArea');
          if (errorEl && errorEl.classList.contains('active')) {
               errorEl.classList.remove('active');
               console.log("Error display area deactivated.");
          }

          if (isSettingsActive && !forceExitSettingsOnErrorHide) {
               console.log("Error hidden, attempting to restore settings view if it was replaced.");
               if (!document.getElementById('settingsContentArea').classList.contains('active')) {
                    switchToView('settingsContentArea');
               }
          } else {
               if (isSettingsActive && forceExitSettingsOnErrorHide) {
                    toggleSettingsMode();
               } else if (!isSettingsActive) {
                    switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView');
               }
          }
     }

     function toggleSettingsMode() {
          isSettingsActive = !isSettingsActive;
          console.log(`Toggling settings mode. isSettingsActive is now: ${isSettingsActive}`);
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

               switchToView('settingsContentArea');

               settingsContentTitle.style.display = 'flex';

               const currentActiveNav = sidebarSettingsNavContainer.querySelector('.settings-nav-btn.active');
               if (currentActiveNav) { displaySettingsTab(currentActiveNav.dataset.targetTab); }
               else if (sidebarSettingsNavContainer.querySelector('.settings-nav-btn')) {
                    sidebarSettingsNavContainer.querySelector('.settings-nav-btn').click();
               }
               settingsBtn.innerHTML = '<i class="fas fa-arrow-left"></i>'; settingsBtn.title = "Back to Main View";
          } else { // Exiting settings mode
               sidebarMainContent.style.display = 'flex';
               sidebarSettingsNavContainer.style.display = 'none';
               settingsContentTitle.style.display = 'none';
               switchToView(originalImageObjectUrl ? 'imageDisplayView' : 'initialView');
               settingsBtn.innerHTML = '<i class="fas fa-cog"></i>'; settingsBtn.title = "Settings";
          }
     }

     function displaySettingsTab(tabId) {
          console.log(`Displaying settings tab: ${tabId}`);
          if (!settingsContentArea) { console.error("settingsContentArea not found!"); return; }

          if (activeSettingsTabContent && activeSettingsTabContent.parentNode === settingsContentArea) {
               settingsContentArea.removeChild(activeSettingsTabContent);
          }
          activeSettingsTabContent = null;

          const tabContentClone = settingsTabsTemplate.querySelector(`#${tabId}`)?.cloneNode(true);

          if (tabContentClone) {
               tabContentClone.classList.remove('active');
               settingsContentArea.appendChild(tabContentClone);
               activeSettingsTabContent = tabContentClone;

               if (tabId === 'settingsLogs') {
                    const fullLogEl = activeSettingsTabContent.querySelector('#fullLogOutput');
                    if (fullLogEl) {
                         fullLogEl.innerHTML = '';
                         currentLogBuffer.forEach(logMsg => appendLogToElement(logMsg, fullLogEl, false));
                         fullLogEl.scrollTop = fullLogEl.scrollHeight;
                    }
               }
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
               if (tabId === 'settingsSystemInfo') {
                    fetchSystemInfo();
               }
          } else {
               console.error(`Settings tab content for ID '${tabId}' not found in template.`);
               settingsContentArea.innerHTML = `<p style="padding: 20px; text-align: center;">Error: Content for tab "${tabId}" could not be loaded.</p>`;
          }
     }

     function fetchSystemInfo() {
          const sysInfoContentEl = activeSettingsTabContent?.querySelector('#systemInfoContent');
          if (!sysInfoContentEl) { console.warn("System info content element not found in active tab."); return; }
          sysInfoContentEl.innerHTML = '<p><i class="fas fa-spinner fa-spin"></i> Loading system info...</p>';
          fetch('/system_info')
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (data.error) { sysInfoContentEl.innerHTML = `<p>Error: ${data.error}</p>`; return; }
                    let html = `<p><strong>App Version:</strong> ${data.app_version || 'N/A'}</p>
                           <p><strong>Python:</strong> ${data.python_version || 'N/A'}</p>
                           <p><strong>PyTorch:</strong> ${data.torch_version || 'N/A'}</p>
                           <p><strong>CUDA:</strong> ${data.cuda_available ? `Yes (Version: ${data.cuda_version || 'N/A'})` : 'No'}</p>`;
                    if (data.gpus && data.gpus.length > 0) html += `<p><strong>GPUs:</strong> ${data.gpus.join(', ')}</p>`;
                    else if (data.cuda_available) html += `<p><strong>GPUs:</strong> (None detected or PyTorch not seeing them)</p>`;
                    html += `<p><strong>OS:</strong> ${data.os || 'N/A'}</p>
                        <p><strong>CPU:</strong> ${data.cpu || 'N/A'}</p>
                        <p><strong>RAM:</strong> Total: ${data.ram?.total || 'N/A'}, Available: ${data.ram?.available || 'N/A'}</p>`;
                    sysInfoContentEl.innerHTML = html;
               }).catch(err => {
                    console.error("Failed to fetch system info:", err);
                    sysInfoContentEl.innerHTML = `<p>Failed to fetch system info: ${err.message}</p>`;
               });
     }
     function handleRestartBackend() {
          const restartStatusEl = activeSettingsTabContent?.querySelector('#restartStatus');
          if (!confirm("Are you sure you want to attempt to restart the backend? This may interrupt ongoing tasks.")) return;
          if (restartStatusEl) restartStatusEl.textContent = "Attempting restart...";
          fetch('/restart_backend', { method: 'POST' })
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (restartStatusEl) restartStatusEl.textContent = data.message || "Restart command sent.";
                    alert((data.message || "Restart command sent.") + (data.success ? "\nThe application might reload or require a manual refresh shortly." : "\nRestart might have failed. Check server logs."));
               }).catch(err => {
                    console.error("Failed to send restart command:", err);
                    if (restartStatusEl) restartStatusEl.textContent = `Error sending command: ${err.message}`;
                    alert("Failed to send restart command to the backend.");
               });
     }
     function loadCurrentOutputDirectory() {
          const customOutputDirInputEl = activeSettingsTabContent?.querySelector('#customOutputDirectory');
          const outputFolderDispEl = activeSettingsTabContent?.querySelector('#defaultOutputFolderDisplay');
          if (!customOutputDirInputEl && !outputFolderDispEl) return;

          if (outputFolderDispEl) outputFolderDispEl.textContent = "Loading...";
          fetch('/get_output_directory')
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (data.output_directory) {
                         if (customOutputDirInputEl) customOutputDirInputEl.value = data.output_directory;
                         if (outputFolderDispEl) outputFolderDispEl.textContent = data.output_directory;
                    } else if (data.error) {
                         if (outputFolderDispEl) outputFolderDispEl.textContent = `Error: ${data.error}`;
                    } else {
                         if (outputFolderDispEl) outputFolderDispEl.textContent = "N/A (Using default)";
                    }
               }).catch(err => {
                    console.error('Error fetching output dir:', err);
                    if (outputFolderDispEl) outputFolderDispEl.textContent = "Error fetching path.";
               });
     }
     function handleSaveOutputDirectory() {
          const customOutputDirInputEl = activeSettingsTabContent?.querySelector('#customOutputDirectory');
          const outputDirStatusEl = activeSettingsTabContent?.querySelector('#outputDirStatus');
          if (!customOutputDirInputEl || !outputDirStatusEl) return;
          const newPath = customOutputDirInputEl.value.trim();
          if (!newPath) { outputDirStatusEl.textContent = 'Path cannot be empty.'; return; }

          outputDirStatusEl.textContent = 'Saving...';
          fetch('/set_output_directory', {
               method: 'POST',
               headers: { 'Content-Type': 'application/json' },
               body: JSON.stringify({ output_directory: newPath })
          })
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (data.error) {
                         outputDirStatusEl.textContent = `Error: ${data.error}`;
                         appendLog(`[ERROR] Set output dir: ${data.error}`);
                    } else {
                         outputDirStatusEl.textContent = data.message || 'Path saved successfully!';
                         const outputFolderDispEl = activeSettingsTabContent?.querySelector('#defaultOutputFolderDisplay');
                         if (outputFolderDispEl && data.new_path) outputFolderDispEl.textContent = data.new_path;
                         appendLog(`[INFO] Output directory set to: ${data.new_path || newPath}`);
                    }
               }).catch(err => {
                    outputDirStatusEl.textContent = 'Error saving path. Check console/backend logs.';
                    appendLog(`[ERROR] Set output dir failed: ${err}`);
                    console.error("Error saving output directory:", err);
               });
     }
     function handleClearBackendDirs() {
          const clearDirsStatusEl = activeSettingsTabContent?.querySelector('#clearDirsStatus');
          if (!confirm("DANGER! This will attempt to delete all images from the WebUI's temporary input and output folders on the server. This action is irreversible. Are you absolutely sure?")) return;
          if (clearDirsStatusEl) clearDirsStatusEl.textContent = 'Clearing directories...';
          fetch('/clear_backend_dirs', { method: 'POST' })
               .then(response => {
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    return response.json();
               })
               .then(data => {
                    if (clearDirsStatusEl) clearDirsStatusEl.textContent = data.message || 'Clear command sent to backend.';
                    appendLog(`[INFO] Clear backend dirs: ${data.message}`);
                    if (data.errors && data.errors.length > 0) {
                         data.errors.forEach(e => appendLog(`[ERROR] Clear backend dirs: ${e}`));
                         alert("Some errors occurred while clearing directories. Check logs.\n" + data.errors.join("\n"));
                    } else if (data.message) {
                         alert(data.message);
                    }
               }).catch(err => {
                    if (clearDirsStatusEl) clearDirsStatusEl.textContent = 'Error sending command.';
                    appendLog(`[ERROR] Clear backend dirs failed: ${err}`);
                    alert("Failed to send clear directories command to backend.");
               });
     }
});