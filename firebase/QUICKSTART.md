# PREBORN Safety System - Quick Start Guide

## 🎯 What's Been Added

Your Firebase project now includes:

### ✅ Login System
- Secure authentication using Firebase Auth
- Professional login page with Preborn branding
- Auto-redirect to dashboard after login
- Sign out functionality

### ✅ Alert Notification System
- Real-time popup alerts when workers enter danger zones
- Alert history panel showing all past incidents
- Sound notifications (browser permitting)
- Visual statistics dashboard

### ✅ Email Notifications
- Automatic email alerts sent to client
- Professional HTML email template
- Includes timestamp and alert details
- Direct link to dashboard

### ✅ Enhanced Dashboard
- Modern UI with Preborn branding (purple gradient theme)
- Real-time statistics cards (Total/Safe/Unsafe Workers, Equipment)
- Improved chart visualization
- User info display in header
- Responsive design

## 🚀 Quick Start (5 Minutes)

### Step 1: Enable Authentication (1 min)
1. Open [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Go to **Authentication** → Click **Get Started**
4. Click on **Email/Password** → Enable it → Save

### Step 2: Create User Account (1 min)
1. In Authentication section → **Users** tab
2. Click **Add User**
3. Enter email: `admin@preborn.com` (or your email)
4. Enter a strong password
5. Click **Add User**

### Step 3: Set Client Email (1 min)
1. Go to **Realtime Database**
2. Click on the root (+)
3. Add this structure:
   ```
   config
     ├─ threshold: 150
     └─ clientEmail: "client@example.com"
   ```
   (Replace with actual client email)

### Step 4: Deploy Database Rules (1 min)
```bash
cd /home/gyandeep_das/Documents/firebases/firebase
firebase deploy --only database
```

### Step 5: Deploy Hosting (1 min)
```bash
firebase deploy --only hosting
```

## 🎉 You're Done! (For Basic Setup)

Visit your site: `https://YOUR-PROJECT-ID.web.app/login.html`

Login with the credentials you created in Step 2.

## 📧 Email Setup (Optional - 10 Minutes)

For email notifications to work, you need to set up Cloud Functions:

### Install Dependencies
```bash
cd functions
npm install
cd ..
```

### Configure Gmail
1. Create a Gmail App Password:
   - Go to [Google Account](https://myaccount.google.com/)
   - Security → 2-Step Verification (enable if not enabled)
   - App Passwords → Select "Mail" and generate
   
2. Configure Firebase Functions:
```bash
firebase functions:config:set gmail.email="your-email@gmail.com"
firebase functions:config:set gmail.password="your-16-char-app-password"
```

### Deploy Functions
```bash
firebase deploy --only functions
```

**Note**: Cloud Functions require the Blaze (pay-as-you-go) plan. It's free for small usage.

## 🧪 Testing the System

1. **Login Test**: Visit login page, enter credentials
2. **Dashboard Test**: Should see the monitoring dashboard
3. **Alert Test**: When Python script detects unsafe worker, you should see:
   - Popup notification (top-right)
   - Updated statistics
   - New entry in Alert History
   - Email sent (if functions deployed)

## 📱 Features Overview

### Login Page (`/login.html`)
- Beautiful gradient background
- Secure authentication
- Error handling
- Auto-redirect when logged in

### Dashboard (`/index.html`)
- **Header**: Preborn logo, user email, sign out button
- **Stats Cards**: Real-time counts of workers and equipment
- **Live Chart**: Visual representation of site (preserved from original)
- **Safety Controls**: Adjustable threshold slider
- **Alert History**: List of all safety incidents

### Notifications
- **Popup Alerts**: Appear in top-right corner
- **Auto-dismiss**: After 10 seconds
- **Email Alerts**: Sent to configured client email

## 🔧 Configuration

### Change Client Email
Update in Firebase Database: `config/clientEmail`

### Adjust Safety Threshold
Use the slider on dashboard (saved automatically)

### Add More Users
Firebase Console → Authentication → Add User

## 📊 Database Structure

```
firebase-database/
├── detections/          # Worker & equipment positions (from your Python script)
├── config/
│   ├── threshold        # Safety distance threshold
│   └── clientEmail      # Email for notifications
├── alerts/              # Alert history
└── emailQueue/          # Email sending queue
```

## 🎨 Customization

All UI colors use the Preborn theme:
- Primary: `#667eea` to `#764ba2` (purple gradient)
- Safe: `#4caf50` (green)
- Danger: `#f44336` (red)
- Info: `#2196f3` (blue)

## ⚠️ Important Notes

- **Original Code Preserved**: Your worker detection logic remains unchanged
- **Security**: All dashboard routes require authentication
- **Real-time**: Everything updates live via Firebase listeners
- **No Breaking Changes**: Existing functionality still works

## 🐛 Troubleshooting

### Can't Login
- Check if Email/Password auth is enabled in Firebase Console
- Verify user account exists
- Check browser console for errors

### No Alerts Showing
- Ensure database rules are deployed
- Check if detections are being written by Python script
- Verify user is authenticated

### Emails Not Sending
- Confirm functions are deployed
- Check Gmail config: `firebase functions:config:get`
- Verify Cloud Functions logs: `firebase functions:log`
- Ensure Blaze plan is active

## 📞 Need Help?

Check the logs:
```bash
# Functions logs
firebase functions:log

# Hosting logs  
firebase hosting:channel:list
```

## 🎓 Next Steps

1. Test the login system
2. Verify alerts are working
3. Set up email notifications
4. Customize client email
5. Monitor the system

---

**Everything is ready! Just follow the Quick Start steps above.** 🚀
