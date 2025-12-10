// Import Firebase Auth functions
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

// Firebase Configuration
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

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Check if user is already logged in
onAuthStateChanged(auth, (user) => {
    if (user) {
        // User is signed in, redirect to dashboard
        window.location.href = 'index.html';
    }
});

// Handle login form submission
const loginForm = document.getElementById('loginForm');
const errorMessage = document.getElementById('errorMessage');

loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    
    // Clear previous error
    errorMessage.style.display = 'none';
    errorMessage.textContent = '';
    
    // Add loading state
    loginForm.classList.add('loading');
    
    try {
        // Sign in with email and password
        await signInWithEmailAndPassword(auth, email, password);
        // Redirect will happen automatically via onAuthStateChanged
    } catch (error) {
        // Handle errors
        loginForm.classList.remove('loading');
        
        let message = 'Login failed. Please try again.';
        
        switch (error.code) {
            case 'auth/invalid-email':
                message = 'Invalid email address format.';
                break;
            case 'auth/user-disabled':
                message = 'This account has been disabled.';
                break;
            case 'auth/user-not-found':
                message = 'No account found with this email.';
                break;
            case 'auth/wrong-password':
                message = 'Incorrect password.';
                break;
            case 'auth/invalid-credential':
                message = 'Invalid email or password.';
                break;
        }
        
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }
});
