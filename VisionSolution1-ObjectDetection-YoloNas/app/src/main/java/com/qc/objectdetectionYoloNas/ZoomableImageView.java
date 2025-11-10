package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import androidx.appcompat.widget.AppCompatImageView;

/**
 * Custom ImageView with pinch-to-zoom and pan functionality
 */
public class ZoomableImageView extends AppCompatImageView {

    private Matrix matrix;
    private Matrix savedMatrix;
    
    // We can be in one of these 3 states
    private static final int NONE = 0;
    private static final int DRAG = 1;
    private static final int ZOOM = 2;
    private int mode = NONE;
    
    // Remember some things for zooming
    private PointF start = new PointF();
    private PointF mid = new PointF();
    private float oldDist = 1f;
    private float[] matrixValues = new float[9];
    
    // Scale gesture detector for pinch-to-zoom
    private ScaleGestureDetector scaleDetector;
    private GestureDetector gestureDetector;
    
    // Zoom constraints
    private static final float MIN_ZOOM = 1f;
    private static final float MAX_ZOOM = 10f;
    
    private OnClickListener externalClickListener;

    public ZoomableImageView(Context context) {
        super(context);
        init(context);
    }

    public ZoomableImageView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    public ZoomableImageView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init(context);
    }

    private void init(Context context) {
        matrix = new Matrix();
        savedMatrix = new Matrix();
        setScaleType(ScaleType.MATRIX);
        setImageMatrix(matrix);
        
        scaleDetector = new ScaleGestureDetector(context, new ScaleListener());
        gestureDetector = new GestureDetector(context, new GestureListener());
        
        setOnTouchListener(new OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                scaleDetector.onTouchEvent(event);
                gestureDetector.onTouchEvent(event);
                
                PointF curr = new PointF(event.getX(), event.getY());
                
                switch (event.getAction() & MotionEvent.ACTION_MASK) {
                    case MotionEvent.ACTION_DOWN:
                        savedMatrix.set(matrix);
                        start.set(curr);
                        mode = DRAG;
                        break;
                        
                    case MotionEvent.ACTION_POINTER_DOWN:
                        oldDist = spacing(event);
                        if (oldDist > 10f) {
                            savedMatrix.set(matrix);
                            midPoint(mid, event);
                            mode = ZOOM;
                        }
                        break;
                        
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_POINTER_UP:
                        mode = NONE;
                        break;
                        
                    case MotionEvent.ACTION_MOVE:
                        if (mode == DRAG) {
                            matrix.set(savedMatrix);
                            float dx = curr.x - start.x;
                            float dy = curr.y - start.y;
                            matrix.postTranslate(dx, dy);
                            
                        } else if (mode == ZOOM) {
                            float newDist = spacing(event);
                            if (newDist > 10f) {
                                matrix.set(savedMatrix);
                                float scale = newDist / oldDist;
                                matrix.postScale(scale, scale, mid.x, mid.y);
                            }
                        }
                        break;
                }
                
                // Apply zoom constraints and boundary checks
                limitZoomAndPan();
                setImageMatrix(matrix);
                return true;
            }
        });
    }

    private class GestureListener extends GestureDetector.SimpleOnGestureListener {
        @Override
        public boolean onSingleTapConfirmed(MotionEvent e) {
            if (externalClickListener != null) {
                externalClickListener.onClick(ZoomableImageView.this);
            }
            return true;
        }

        @Override
        public boolean onDoubleTap(MotionEvent e) {
            // Double tap to reset zoom
            resetZoom();
            return true;
        }
    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            float scaleFactor = detector.getScaleFactor();
            
            matrix.postScale(scaleFactor, scaleFactor, detector.getFocusX(), detector.getFocusY());
            limitZoomAndPan();
            setImageMatrix(matrix);
            
            return true;
        }
    }

    private void limitZoomAndPan() {
        matrix.getValues(matrixValues);
        float scaleX = matrixValues[Matrix.MSCALE_X];
        float scaleY = matrixValues[Matrix.MSCALE_Y];
        float scale = Math.min(scaleX, scaleY);
        
        // Limit zoom level
        if (scale < MIN_ZOOM) {
            matrix.setScale(MIN_ZOOM, MIN_ZOOM);
        } else if (scale > MAX_ZOOM) {
            matrix.postScale(MAX_ZOOM / scale, MAX_ZOOM / scale);
        }
        
        // Center the image if it's smaller than the view
        matrix.getValues(matrixValues);
        float currentScale = matrixValues[Matrix.MSCALE_X];
        
        if (getDrawable() != null) {
            float imageWidth = getDrawable().getIntrinsicWidth() * currentScale;
            float imageHeight = getDrawable().getIntrinsicHeight() * currentScale;
            float viewWidth = getWidth();
            float viewHeight = getHeight();
            
            float deltaX = 0, deltaY = 0;
            
            if (imageWidth <= viewWidth) {
                deltaX = (viewWidth - imageWidth) / 2 - matrixValues[Matrix.MTRANS_X];
            } else if (matrixValues[Matrix.MTRANS_X] > 0) {
                deltaX = -matrixValues[Matrix.MTRANS_X];
            } else if (matrixValues[Matrix.MTRANS_X] < viewWidth - imageWidth) {
                deltaX = viewWidth - imageWidth - matrixValues[Matrix.MTRANS_X];
            }
            
            if (imageHeight <= viewHeight) {
                deltaY = (viewHeight - imageHeight) / 2 - matrixValues[Matrix.MTRANS_Y];
            } else if (matrixValues[Matrix.MTRANS_Y] > 0) {
                deltaY = -matrixValues[Matrix.MTRANS_Y];
            } else if (matrixValues[Matrix.MTRANS_Y] < viewHeight - imageHeight) {
                deltaY = viewHeight - imageHeight - matrixValues[Matrix.MTRANS_Y];
            }
            
            matrix.postTranslate(deltaX, deltaY);
        }
    }

    private float spacing(MotionEvent event) {
        if (event.getPointerCount() < 2) return 0;
        
        float x = event.getX(0) - event.getX(1);
        float y = event.getY(0) - event.getY(1);
        return (float) Math.sqrt(x * x + y * y);
    }

    private void midPoint(PointF point, MotionEvent event) {
        if (event.getPointerCount() < 2) return;
        
        float x = event.getX(0) + event.getX(1);
        float y = event.getY(0) + event.getY(1);
        point.set(x / 2, y / 2);
    }

    @Override
    public void setOnClickListener(OnClickListener l) {
        this.externalClickListener = l;
    }

    public void resetZoom() {
        matrix.reset();
        setImageMatrix(matrix);
    }

    @Override
    public void setImageDrawable(Drawable drawable) {
        super.setImageDrawable(drawable);
        resetZoom();
    }
}