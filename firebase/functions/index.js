const functions = require('firebase-functions');
const admin = require('firebase-admin');
const nodemailer = require('nodemailer');

admin.initializeApp();

// Configure email transporter (you'll need to set these environment variables)
// Firebase CLI: firebase functions:config:set gmail.email="youremail@gmail.com" gmail.password="yourapppassword"
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: functions.config().gmail?.email || process.env.GMAIL_EMAIL,
        pass: functions.config().gmail?.password || process.env.GMAIL_PASSWORD
    }
});

// Send email when a new email request is added to the queue
exports.sendAlertEmail = functions.database.ref('/emailQueue/{pushId}')
    .onCreate(async (snapshot, context) => {
        const emailData = snapshot.val();
        
        if (emailData.processed) {
            return null;
        }
        
        // Generate HTML based on email type
        const isLoginEmail = emailData.type === 'login';
        const emailHtml = isLoginEmail ? `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #f5f5f5;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 2em; letter-spacing: 1px;"><span style="font-weight: 800;">Pre</span><span style="font-weight: 600;">Born</span></h1>
                    <p style="color: white; margin: 10px 0 0 0; font-size: 14px;">Safety Monitoring System</p>
                </div>
                
                <div style="background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 25px; margin: 0;">
                    <h2 style="color: #1976d2; margin-top: 0; display: flex; align-items: center; gap: 10px;">🔐 New Login Detected</h2>
                    <p style="font-size: 16px; line-height: 1.6; color: #333;">
                        <strong>Time:</strong> ${new Date(emailData.timestamp).toLocaleString()}<br>
                        <strong>Email:</strong> ${emailData.to}<br>
                        ${emailData.loginInfo ? `<strong>Device:</strong> ${emailData.loginInfo.platform}<br>
                        <strong>Timezone:</strong> ${emailData.loginInfo.location}<br>` : ''}
                    </p>
                </div>
                
                <div style="padding: 30px; background-color: white;">
                    <h3 style="color: #333; margin-top: 0;">Security Notice</h3>
                    <p style="color: #666; line-height: 1.6;">A new login to your PreBorn Safety Dashboard has been detected. If this was you, no action is needed.</p>
                    <p style="color: #666; line-height: 1.6;"><strong>If you didn't log in:</strong> Please secure your account immediately and change your password.</p>
                    <a href="https://worker-detection-and-safety.web.app/login.html" 
                       style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 35px; text-decoration: none; border-radius: 8px; margin-top: 15px; font-weight: 600;">
                        Access Dashboard
                    </a>
                </div>
                
                <div style="padding: 20px; text-align: center; color: #999; font-size: 12px; background-color: #f5f5f5;">
                    <p style="margin: 5px 0;">This is an automated security notification from PreBorn Safety Monitoring System.</p>
                    <p style="margin: 5px 0;">© ${new Date().getFullYear()} PreBorn. All rights reserved.</p>
                </div>
            </div>
        ` : `
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #f5f5f5;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 2em; letter-spacing: 1px;"><span style="font-weight: 800;">Pre</span><span style="font-weight: 600;">Born</span></h1>
                    <p style="color: white; margin: 10px 0 0 0; font-size: 14px;">Safety Monitoring System</p>
                </div>
                
                <div style="background-color: #fff3cd; border-left: 5px solid #ff9800; padding: 25px; margin: 0;">
                    <h2 style="color: #f57c00; margin-top: 0; display: flex; align-items: center; gap: 10px;">⚠️ Safety Alert</h2>
                    <p style="font-size: 16px; line-height: 1.6; color: #333;">
                        <strong>Time:</strong> ${new Date(emailData.timestamp).toLocaleString()}<br>
                        <strong>Alert:</strong> ${emailData.message}
                    </p>
                </div>
                
                <div style="padding: 30px; background-color: white;">
                    <h3 style="color: #d32f2f; margin-top: 0;">⚡ Immediate Action Required</h3>
                    <p style="color: #666; line-height: 1.6;">A safety hazard has been detected on your construction site. Please check the dashboard immediately for more details and take necessary action.</p>
                    <a href="https://worker-detection-and-safety.web.app" 
                       style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 35px; text-decoration: none; border-radius: 8px; margin-top: 15px; font-weight: 600; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);">
                        View Dashboard Now
                    </a>
                </div>
                
                <div style="padding: 20px; text-align: center; color: #999; font-size: 12px; background-color: #f5f5f5;">
                    <p style="margin: 5px 0;">This is an automated message from PreBorn Safety Monitoring System.</p>
                    <p style="margin: 5px 0;">© ${new Date().getFullYear()} PreBorn. All rights reserved.</p>
                </div>
            </div>
        `;
        
        const mailOptions = {
            from: `"PreBorn Safety System" <${functions.config().gmail?.email || process.env.GMAIL_EMAIL}>`,
            to: emailData.to,
            subject: emailData.subject,
            text: emailData.message,
            html: emailHtml
        };
        
        try {
            await transporter.sendMail(mailOptions);
            console.log('Email sent successfully to:', emailData.to);
            
            // Mark as processed
            await snapshot.ref.update({ processed: true, sentAt: admin.database.ServerValue.TIMESTAMP });
            
            return null;
        } catch (error) {
            console.error('Error sending email:', error);
            await snapshot.ref.update({ processed: false, error: error.message });
            return null;
        }
    });

// Optional: Clean up old processed emails (runs daily)
exports.cleanupEmailQueue = functions.pubsub.schedule('every 24 hours').onRun(async (context) => {
    const db = admin.database();
    const emailQueueRef = db.ref('/emailQueue');
    
    const snapshot = await emailQueueRef.once('value');
    const emails = snapshot.val();
    
    if (!emails) return null;
    
    const now = Date.now();
    const updates = {};
    
    Object.keys(emails).forEach(key => {
        const email = emails[key];
        if (email.processed && email.sentAt) {
            // Delete emails older than 7 days
            if (now - email.sentAt > 7 * 24 * 60 * 60 * 1000) {
                updates[key] = null;
            }
        }
    });
    
    if (Object.keys(updates).length > 0) {
        await emailQueueRef.update(updates);
        console.log(`Cleaned up ${Object.keys(updates).length} old emails`);
    }
    
    return null;
});
