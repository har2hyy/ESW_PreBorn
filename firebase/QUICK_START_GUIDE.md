# Quick Start Guide: New Features

## 🌓 Theme Toggle

### Location
Look for this button in the top-right header area:

```
┌─────────────────────────────────────────────────────┐
│  PreBorn  [🌙 Dark] [👤 user@email.com] [Sign Out] │
│                 ↑                                    │
│           Theme Toggle                              │
└─────────────────────────────────────────────────────┘
```

### How to Use
1. Click the theme button (shows 🌙 moon in light mode, ☀️ sun in dark mode)
2. Dashboard instantly switches theme
3. Your choice is saved automatically
4. Works across all pages

### Visual Comparison

**Light Mode** (Default):
- Clean white cards
- Dark text on light background
- Professional daytime appearance

**Dark Mode**:
- Dark blue-gray cards
- Light text on dark background
- Reduced eye strain for nighttime use

---

## 📊 3D Depth Visualization

### Location
Scroll down to find the new "3D Depth Visualization" section between the 2D chart and controls:

```
┌──────────────────────────────┐
│  Live Site Monitoring (2D)   │
│  [Scatter plot chart]        │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  3D Depth Visualization      │  ← NEW!
│  [Interactive 3D surface]    │
└──────────────────────────────┘
           ↓
┌──────────────────────────────┐
│  Safety Distance Threshold   │
│  [Slider control]            │
└──────────────────────────────┘
```

### Interactive Controls

**Rotate View**:
- Click and hold left mouse button
- Drag in any direction
- Release to stop rotating

**Zoom In/Out**:
- Scroll mouse wheel up (zoom in)
- Scroll mouse wheel down (zoom out)
- Or use pinch gesture on trackpad

**View Data**:
- Hover cursor over any point on surface
- Tooltip shows X, Y, Z coordinates
- Color indicates depth level (blue=deep, yellow=shallow)

### Understanding the Chart

**Axes**:
- **X-axis**: Horizontal position (pixels)
- **Y-axis**: Vertical position (pixels)
- **Z-axis**: Depth value (relative distance)

**Colors**:
- **Dark Blue/Purple**: Greater depth (farther away)
- **Green/Yellow**: Lesser depth (closer)
- **Colorbar**: Shows scale on right side

---

## 🎮 Keyboard Shortcuts

Currently implemented:
- **Click Theme Button**: Toggle light/dark mode

Planned shortcuts:
- `Ctrl + D`: Toggle dark mode
- `R`: Reset 3D view
- `F`: Toggle fullscreen 3D chart

---

## 💡 Tips & Tricks

### Theme Toggle
1. **Eye Comfort**: Use dark mode in low-light environments
2. **Screenshots**: Switch to light mode for clearer printouts
3. **Presentation**: Dark mode looks professional in demos

### 3D Visualization
1. **Best View Angle**: Start with default, then rotate to see depth variations
2. **Zoom Smart**: Zoom in to see individual detection points
3. **Find Patterns**: Rotate slowly to identify depth clusters
4. **Compare**: Use alongside 2D chart for comprehensive view

### Performance
1. **Smooth Interaction**: Close unused browser tabs
2. **Data Updates**: 3D chart refreshes when new detections arrive
3. **Theme Switch**: Instant - no lag or reload needed

---

## 🔧 Troubleshooting

### Theme Toggle Issues

**Problem**: Theme doesn't change
- **Solution**: Refresh page (Ctrl+R or F5)
- **Check**: Browser supports localStorage

**Problem**: Theme resets after closing browser
- **Solution**: Check browser isn't in private/incognito mode
- **Note**: Private browsing doesn't save preferences

### 3D Chart Issues

**Problem**: Chart shows "No data available"
- **Solution**: Wait for detection data to load
- **Check**: Firebase connection is active

**Problem**: Chart is laggy or slow
- **Solution**: Reduce browser zoom level
- **Try**: Close other tabs, restart browser

**Problem**: Can't rotate or zoom
- **Solution**: Click inside chart area first
- **Check**: Mouse/trackpad is working properly

---

## 📞 Support

### Feature Requests
Want a new theme color or chart type? Contact the admin.

### Bug Reports
If something doesn't work:
1. Note the error message (if any)
2. Take a screenshot
3. Describe what you were doing
4. Report to admin with details

### Documentation
For developers:
- See `THEME_AND_3D_FEATURES.md` for technical details
- Check `index.html` for theme implementation
- Review `app.js` for 3D chart integration

---

## 🎯 Best Practices

### Daily Usage
1. **Start Session**:
   - Log in
   - Choose preferred theme
   - Check 2D chart for overview

2. **Monitor Site**:
   - Watch live 2D scatter plot
   - Check 3D depth map periodically
   - Review stats cards

3. **Investigate Alerts**:
   - Use 2D chart for quick location
   - Switch to 3D for depth analysis
   - Review alert history

### Team Collaboration
1. **Screenshots**: Use light mode for sharing
2. **Presentations**: Dark mode for screen sharing
3. **Reports**: Include both 2D and 3D views

---

## 🚀 What's Next?

Coming soon:
- [ ] Auto-theme (follows system settings)
- [ ] Custom color schemes
- [ ] Real depth sensor data integration
- [ ] Export 3D visualization as image
- [ ] Time-lapse depth animation
- [ ] Side-by-side 2D/3D view
- [ ] Mobile-optimized 3D controls

---

## ✨ Quick Reference

| Action | Method |
|--------|--------|
| Switch Theme | Click 🌙/☀️ button |
| Rotate 3D | Click + Drag |
| Zoom 3D | Scroll Wheel |
| Reset 3D | Double Click |
| View Depth | Hover on surface |

---

**Dashboard URL**: https://worker-detection-and-safety.web.app

**Last Updated**: [Current Session]

**Features Status**: ✅ Live and Deployed
