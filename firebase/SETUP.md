# PREBORN Safety Monitoring System

Complete setup with login system, alert notifications, and email functionality.

## 🎯 Features

- **User Authentication**: Secure login system using Firebase Authentication
- **Real-time Monitoring**: Live worker and equipment detection dashboard
- **Alert System**: Instant notifications when workers enter danger zones
- **Email Notifications**: Automatic email alerts sent to client
- **Alert History**: View past safety incidents
- **User-Friendly Interface**: Modern, responsive design with Preborn branding

## 📋 Setup Instructions

### 1. Enable Firebase Authentication

```bash
# In Firebase Console:
# 1. Go to Authentication section
# 2. Click "Get Started"
# 3. Enable "Email/Password" sign-in method
```

### 2. Create a User Account

```bash
# In Firebase Console Authentication section:
# Click "Add User" and create an account with:
# Email: your-email@example.com
# Password: (choose a secure password)
```

### 3. Set Client Email for Notifications

Open Firebase Realtime Database and add this data:

```json
{
  "config": {
    "threshold": 150,
    "clientEmail": "client@example.com"
  }
}
```

### 4. Setup Email Notifications (Cloud Functions)

```bash
# Navigate to functions directory
cd functions

# Install dependencies
npm install

# Configure Gmail credentials (use App Password, not regular password)
firebase functions:config:set gmail.email="your-email@gmail.com" gmail.password="your-app-password"

# Deploy functions
cd ..
firebase deploy --only functions
```

**To create Gmail App Password:**
1. Go to Google Account Settings → Security
2. Enable 2-Step Verification
3. Go to App Passwords
4. Generate a new app password for "Mail"
5. Use this password in the config above

### 5. Deploy the Application

```bash
# Deploy database rules
firebase deploy --only database

# Deploy hosting
firebase deploy --only hosting

# Or deploy everything
firebase deploy
```

### 6. Update Database Rules

The database rules have been updated to allow authenticated read/write access. Deploy them:

```bash
firebase deploy --only database
```

## 🚀 Access the Application

1. **Login Page**: `https://your-project.web.app/login.html`
2. **Dashboard**: `https://your-project.web.app/`

Default user credentials (you created in step 2):
- Email: your-email@example.com
- Password: (your chosen password)

## 📧 Email Notifications

Email notifications are automatically sent when:
- Workers enter danger zones
- Unsafe conditions are detected

Emails include:
- Timestamp of alert
- Number of workers in danger
- Direct link to dashboard
- Professional Preborn branding

## 🎨 Dashboard Features

### Statistics Cards
- Total Workers
- Safe Workers
- Unsafe Workers (⚠️ highlighted in red)
- Active Equipment

### Live Monitoring Chart
- Green dots: Safe workers
- Red dots: Unsafe workers (in danger zone)
- Blue dots: Machinery/Equipment

### Alert System
- Real-time popup notifications
- Alert history with timestamps
- Email notification status

### Safety Controls
- Adjustable safety distance threshold
- Real-time threshold updates

## 🔐 Security Notes

- All routes require authentication
- Database rules enforce auth checks
- Sensitive data protected
- Email queue processed securely via Cloud Functions

## 📱 User Flow

1. User visits site → redirected to login
2. User logs in → redirected to dashboard
3. Dashboard shows real-time worker positions
4. When worker enters danger zone:
   - Alert popup appears
   - Alert saved to history
   - Email sent to client
5. User can adjust safety threshold
6. User can sign out anytime

## 🛠️ Files Structure

```
firebase/
├── public/
│   ├── index.html          # Main dashboard (enhanced with Preborn branding)
│   ├── app.js              # Dashboard logic with alerts & email
│   ├── login.html          # Login page
│   ├── login.js            # Login authentication
│   └── alerts.css          # Alert notification styles
├── functions/
│   ├── index.js            # Cloud Functions for email
│   ├── package.json        # Function dependencies
│   └── .gitignore
├── firebase.json           # Firebase configuration
├── database.rules.json     # Updated security rules
└── SETUP.md               # This file
```

## 🎨 Branding

- Company Name: **PREBORN**
- Color Scheme: Purple gradient (#667eea to #764ba2)
- Modern, professional design
- Responsive layout for all devices

## 📞 Support

For issues or questions, check:
1. Firebase Console for error logs
2. Browser console for client errors
3. Functions logs: `firebase functions:log`

## 🔄 Maintaining the System

### View Logs
```bash
firebase functions:log
```

### Update Client Email
Update in Firebase Database: `config/clientEmail`

### View Alerts
Check Firebase Database: `alerts/` path

### Clean Email Queue
Automatic cleanup runs daily (7-day retention)

---

**Note**: The original worker detection functionality remains unchanged. All new features are additions that enhance the existing system.
