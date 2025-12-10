#!/bin/bash

echo "🚀 PREBORN Safety System - Quick Setup"
echo "======================================"
echo ""

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI is not installed."
    echo "Install it with: npm install -g firebase-tools"
    exit 1
fi

echo "✅ Firebase CLI found"
echo ""

# Step 1: Setup Functions
echo "📦 Step 1: Installing Cloud Functions dependencies..."
cd functions
npm install
cd ..
echo "✅ Dependencies installed"
echo ""

# Step 2: Instructions for manual steps
echo "📋 Step 2: Manual Configuration Required"
echo "=========================================="
echo ""
echo "Please complete these steps in Firebase Console:"
echo ""
echo "1. Enable Authentication:"
echo "   - Go to Firebase Console → Authentication"
echo "   - Click 'Get Started'"
echo "   - Enable 'Email/Password' sign-in method"
echo ""
echo "2. Create User Account:"
echo "   - In Authentication section, click 'Add User'"
echo "   - Email: your-email@example.com"
echo "   - Password: (choose a secure password)"
echo ""
echo "3. Set Client Email in Database:"
echo "   - Go to Realtime Database"
echo "   - Add: config/clientEmail = 'client@example.com'"
echo ""
echo "4. Configure Email (Gmail):"
echo "   - Create App Password: https://myaccount.google.com/apppasswords"
echo "   - Run: firebase functions:config:set gmail.email='your@gmail.com' gmail.password='app-password'"
echo ""

read -p "Press Enter when you've completed the above steps..."

# Step 3: Deploy
echo ""
echo "🚀 Step 3: Deploying to Firebase..."
echo ""

read -p "Deploy database rules? (y/n): " deploy_db
if [ "$deploy_db" = "y" ]; then
    firebase deploy --only database
    echo "✅ Database rules deployed"
fi

read -p "Deploy cloud functions? (y/n): " deploy_functions
if [ "$deploy_functions" = "y" ]; then
    firebase deploy --only functions
    echo "✅ Functions deployed"
fi

read -p "Deploy hosting? (y/n): " deploy_hosting
if [ "$deploy_hosting" = "y" ]; then
    firebase deploy --only hosting
    echo "✅ Hosting deployed"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "Your PREBORN Safety System is ready!"
echo ""
echo "📱 Access your application:"
echo "   Login: https://YOUR-PROJECT-ID.web.app/login.html"
echo "   Dashboard: https://YOUR-PROJECT-ID.web.app/"
echo ""
echo "📖 See SETUP.md for detailed documentation"
echo ""
