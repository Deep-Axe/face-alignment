import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections

# Choose your detector - switch between 'sfd', 'mediapipe', 'blazeface', etc.
face_detector = 'sfd'  # Change this to test different detectors

# Set detector-specific parameters
if face_detector == 'sfd':
    face_detector_kwargs = {"filter_threshold": 0.8}
elif face_detector == 'mediapipe':
    face_detector_kwargs = {
        "min_detection_confidence": 0.6,
        "max_num_faces": 1
    }
elif face_detector == 'blazeface':
    face_detector_kwargs = {}
else:
    face_detector_kwargs = {}

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

try:
    input_img = io.imread(r"F:\Thermal_data_test_scripts\IRIS\the_dataset\Balage\Expression\exp2\V-884.bmp")
except FileNotFoundError:
    input_img = io.imread('test/assets/aflw-test.jpg')

# Get standard landmarks (68 points for most detectors)
preds = fa.get_landmarks(input_img)
if not preds:
    print("No faces detected!")
    exit()

preds_68 = preds[-1]  # Get the last (or only) detected face
print(f"Standard landmarks shape: {preds_68.shape}")

# Try to get 478 landmarks if using MediaPipe
mediapipe_landmarks = None
if face_detector == 'mediapipe' and hasattr(fa.face_detector, 'get_landmarks_478'):
    mediapipe_landmarks = fa.face_detector.get_landmarks_478(input_img)
    if mediapipe_landmarks:
        preds_478 = mediapipe_landmarks[0]
        print(f"MediaPipe 478 landmarks shape: {preds_478.shape}")

# Plotting
if mediapipe_landmarks and face_detector == 'mediapipe':
    # Plot both 68 and 478 landmarks for MediaPipe
    fig = plt.figure(figsize=(20, 8))
    
    # Standard 68 landmarks (styled)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(input_img)
    
    # Define landmark groups for 68 points
    plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {
        'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
    }
    
    for pred_type in pred_types.values():
        ax1.plot(preds_68[pred_type.slice, 0],
                preds_68[pred_type.slice, 1],
                color=pred_type.color, **plot_style)
    ax1.set_title(f'{face_detector.upper()} - 68 Landmarks (Styled)')
    ax1.axis('off')
    
    # MediaPipe 478 landmarks (2D)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(input_img)
    ax2.scatter(preds_478[:, 0], preds_478[:, 1], s=1, c='red', alpha=0.6)
    ax2.set_title('MediaPipe - 478 Landmarks (2D)')
    ax2.axis('off')
    
    # MediaPipe 478 landmarks (3D)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(preds_478[:, 0], preds_478[:, 1], preds_478[:, 2], s=1, c='red', alpha=0.6)
    ax3.set_title('MediaPipe - 478 Landmarks (3D)')
    # Add axis labels for 3D plot
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.set_zlabel('Z (depth)')
    
else:
    # Standard plotting for other detectors (original style)
    plot_style = dict(marker='o', markersize=4, linestyle='-', lw=2)
    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {
        'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
    }

    fig = plt.figure(figsize=plt.figaspect(.5))
    
    # 2D Plot
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(input_img)
    for pred_type in pred_types.values():
        ax.plot(preds_68[pred_type.slice, 0],
                preds_68[pred_type.slice, 1],
                color=pred_type.color, **plot_style)
    ax.set_title(f'{face_detector.upper()} - 68 Landmarks (2D)')
    ax.axis('off')

    # 3D Plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.scatter(preds_68[:, 0] * 1.2,
                      preds_68[:, 1],
                      preds_68[:, 2],
                      c='cyan',
                      alpha=1.0,
                      edgecolor='b')

    for pred_type in pred_types.values():
        ax.plot3D(preds_68[pred_type.slice, 0] * 1.2,
                  preds_68[pred_type.slice, 1],
                  preds_68[pred_type.slice, 2], color='blue')

    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_title(f'{face_detector.upper()} - 68 Landmarks (3D)')
    # Add axis labels for 3D plot
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Z (depth)')


    ax.view_init(elev=90., azim=90.)
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_title(f'{face_detector.upper()} - 68 Landmarks (3D)')

plt.tight_layout()
plt.show()

print(f"\nUsed detector: {face_detector}")
print(f"Detected {len(preds)} face(s)")
if mediapipe_landmarks:
    print(f"MediaPipe provided {len(mediapipe_landmarks)} face(s) with 478 landmarks each")