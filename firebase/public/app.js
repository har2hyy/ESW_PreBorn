// --- 1. Import the necessary functions ---
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getDatabase, ref, onValue, set, push } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-database.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

// --- 2. Your Firebase Configuration (this part is correct) ---
const firebaseConfig = {
    apiKey: "AIzaSyBIbJL-3rV0hN1ZlcJRFyTiGgNvNo3FVt4",
    authDomain: "worker-detection-and-safety.firebaseapp.com",
    databaseURL: "https://worker-detection-and-safety-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "worker-detection-and-safety",
    storageBucket: "worker-detection-and-safety.appspot.com",
    messagingSenderId: "897703055939",
    appId: "1:897703055939:web:e9afaf6b7e2d6e55777af6",
    measurementId: "G-FLYZ2HJMZK"
};

// --- 3. Initialize Firebase ---
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);
const detectionsRef = ref(database, 'detections');
const thresholdRef = ref(database, 'config/threshold');
const alertsRef = ref(database, 'alerts');
const clientEmailRef = ref(database, 'config/clientEmail');

// --- 3.1. Authentication Check ---
let currentUser = null;
const loadingOverlay = document.getElementById('loadingOverlay');

onAuthStateChanged(auth, (user) => {
    if (user) {
        currentUser = user;
        document.getElementById('userEmail').textContent = user.email;
        loadingOverlay.style.display = 'none';
        
        // Send login notification email
        sendLoginNotification(user.email);
    } else {
        // User is signed out, redirect to login
        window.location.href = 'login.html';
    }
});

// Logout functionality
document.getElementById('logoutBtn').addEventListener('click', async () => {
    await signOut(auth);
});

// --- 4. Chart.js Setup ---
const ctx = document.getElementById('myChart').getContext('2d');
const myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [{
            label: 'Safe Worker',
            data: [], // Initially empty
            backgroundColor: '#006400'
        }, {
            label: 'Unsafe Worker',
            data: [], // Initially empty
            backgroundColor: '#ff0000',
            pointRadius: 10,
        }, {
            label: 'Machinery',
            data: [], // Initially empty
            backgroundColor: '#0000ff',
            pointRadius: 6,
        }]
    },
    options: {
        scales: {
            x: { type: 'linear', position: 'bottom', min: 0, max: 1920 },
            y: { min: 0, max: 1080, reverse: true }
        }
    }
});

// --- 5. Add Logic for Firebase and Slider ---

// Variables for alert tracking
let lastUnsafeCount = 0;
let alertHistory = [];
let clientEmail = '';

// Get client email for notifications
onValue(clientEmailRef, (snapshot) => {
    clientEmail = snapshot.val() || '';
});

// This function runs whenever the data in '/detections' changes
onValue(detectionsRef, (snapshot) => {
    const detections = snapshot.val();
    const safePoints = [];
    const unsafePoints = [];
    const machineryPoints = [];
    
    if (detections) {
        detections.forEach(obj => {
            const point = { x: obj.centerX, y: obj.centerY };
            
            if (obj.label === "worker" && obj.isUnsafe) {
                unsafePoints.push(point);
            } 
            else if (obj.label === "worker" && !obj.isUnsafe) {
                safePoints.push(point);
            } 
            else {
                machineryPoints.push(point);
            }
        });
    }
    
    myChart.data.datasets[0].data = safePoints;     // Safe Worker (green)
    myChart.data.datasets[1].data = unsafePoints;   // Unsafe Worker (red)
    myChart.data.datasets[2].data = machineryPoints; // Machinery (blue)
    
    myChart.update();
    
    // Update stats
    updateStats(safePoints.length, unsafePoints.length, machineryPoints.length);
    
    // Check for new unsafe workers and trigger alert
    if (unsafePoints.length > lastUnsafeCount && unsafePoints.length > 0) {
        const newUnsafeWorkers = unsafePoints.length - lastUnsafeCount;
        triggerAlert(newUnsafeWorkers, unsafePoints.length);
    }
    lastUnsafeCount = unsafePoints.length;
});

// Update statistics
function updateStats(safe, unsafe, equipment) {
    document.getElementById('totalWorkers').textContent = safe + unsafe;
    document.getElementById('safeWorkers').textContent = safe;
    document.getElementById('unsafeWorkers').textContent = unsafe;
    document.getElementById('totalEquipment').textContent = equipment;
}

// Trigger alert notification
function triggerAlert(newCount, totalUnsafe) {
    const timestamp = new Date().toISOString();
    const message = `${totalUnsafe} worker${totalUnsafe > 1 ? 's are' : ' is'} in danger zone! Immediate action required.`;
    
    // Show popup notification
    showAlertNotification('⚠️ Safety Alert', message);
    
    // Save to database and send email
    const alertData = {
        timestamp: timestamp,
        unsafeCount: totalUnsafe,
        message: message,
        emailSent: false
    };
    
    // Push alert to database
    const newAlertRef = push(alertsRef);
    set(newAlertRef, alertData).then(() => {
        // Trigger email notification
        if (clientEmail) {
            sendEmailNotification(clientEmail, alertData);
        }
    });
}

// Show alert notification popup
function showAlertNotification(title, message) {
    const container = document.getElementById('alertContainer');
    const alert = document.createElement('div');
    alert.className = 'alert';
    
    alert.innerHTML = `
        <div class="alert-icon">⚠️</div>
        <div class="alert-content">
            <div class="alert-title">${title}</div>
            <div class="alert-message">${message}</div>
            <div class="alert-time">${new Date().toLocaleTimeString()}</div>
        </div>
        <button class="alert-close" onclick="this.parentElement.remove()">×</button>
    `;
    
    container.insertBefore(alert, container.firstChild);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (alert.parentElement) {
            alert.remove();
        }
    }, 10000);
    
    // Play alert sound (optional)
    try {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjeR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+Dyvm==');
        audio.play().catch(() => {});
    } catch (e) {}
}

// Send email notification via database trigger
async function sendEmailNotification(email, alertData) {
    // Store email request in database - this would be picked up by a Cloud Function
    const emailRef = ref(database, 'emailQueue');
    const newEmailRef = push(emailRef);
    
    await set(newEmailRef, {
        to: email,
        subject: '⚠️ PreBorn Safety Alert - Immediate Action Required',
        message: `
Safety Alert Detected!

Time: ${new Date(alertData.timestamp).toLocaleString()}
Alert: ${alertData.message}

Please check the dashboard immediately for more details.

This is an automated message from PreBorn Safety Monitoring System.
        `,
        timestamp: alertData.timestamp,
        type: 'alert',
        processed: false
    });
}

// Send login notification email
async function sendLoginNotification(userEmail) {
    const emailRef = ref(database, 'emailQueue');
    const newEmailRef = push(emailRef);
    
    const timestamp = new Date().toISOString();
    const loginInfo = {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        location: Intl.DateTimeFormat().resolvedOptions().timeZone
    };
    
    await set(newEmailRef, {
        to: userEmail,
        subject: '🔐 PreBorn Dashboard - New Login Detected',
        message: `
New Login Alert

A new login to your PreBorn Safety Dashboard has been detected.

Time: ${new Date(timestamp).toLocaleString()}
Email: ${userEmail}
Device: ${loginInfo.platform}
Browser: ${loginInfo.userAgent.split('(')[0]}
Timezone: ${loginInfo.location}

If this wasn't you, please secure your account immediately.

This is an automated security notification from PreBorn Safety Monitoring System.
        `,
        timestamp: timestamp,
        type: 'login',
        loginInfo: loginInfo,
        processed: false
    });
}

// Load and display alert history
onValue(alertsRef, (snapshot) => {
    const alerts = snapshot.val();
    alertHistory = [];
    
    if (alerts) {
        Object.keys(alerts).forEach(key => {
            alertHistory.push({ id: key, ...alerts[key] });
        });
        
        // Sort by timestamp (newest first)
        alertHistory.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        // Display alerts
        displayAlertHistory();
    }
});

function displayAlertHistory() {
    const alertList = document.getElementById('alertList');
    const alertCount = document.getElementById('alertCount');
    
    alertCount.textContent = alertHistory.length;
    
    if (alertHistory.length === 0) {
        alertList.innerHTML = `
            <div class="no-alerts">
                <div class="no-alerts-icon">✓</div>
                <div>No alerts yet. System is monitoring...</div>
            </div>
        `;
        return;
    }
    
    alertList.innerHTML = alertHistory.slice(0, 20).map(alert => {
        const date = new Date(alert.timestamp);
        return `
            <div class="alert-item">
                <div class="alert-item-header">
                    <div class="alert-item-title">Safety Alert</div>
                    <div class="alert-item-time">${date.toLocaleString()}</div>
                </div>
                <div class="alert-item-message">${alert.message}</div>
                ${clientEmail ? '<div class="email-status">Email notification sent</div>' : ''}
            </div>
        `;
    }).join('');
}

// --- 6. Slider Logic ---

// Get the slider and the text element from the HTML
const slider = document.getElementById('thresholdSlider');
const sliderValueDisplay = document.getElementById('thresholdValue');

// Keep slider synced with database
onValue(thresholdRef, (snapshot) => {
    const currentThreshold = snapshot.val();
    if (currentThreshold !== null) {
        slider.value = currentThreshold;
        sliderValueDisplay.textContent = `${currentThreshold}px`;
    }
});

// Update database when slider moves
slider.addEventListener('input', (event) => {
    const newThreshold = event.target.value;
    sliderValueDisplay.textContent = `${newThreshold}px`;
    set(thresholdRef, parseFloat(newThreshold));
});
