import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import io
import base64
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="3D Mondrian Room Planner with Wall Gaps",
    page_icon="🏠",
    layout="wide"
)

# ==================== SOCIAL SHARING ====================
st.markdown("""
    <meta property="og:title" content="3D Mondrian Room Planner - Wall Gap Edition">
    <meta property="og:description" content="Interactive 3D room partitioning with physical wall gaps for architectural planning">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://mondrian-planner.streamlit.app">
    <meta property="og:image" content="https://raw.githubusercontent.com/yusdesign/mondrian-planner/main/og_image.png">
    <meta name="twitter:card" content="summary_large_image">
""", unsafe_allow_html=True)

st.title("🏠 3D Mondrian Room Planner")
st.markdown("### *Recursive axis-aligned room partitioning with real wall gaps*")

# ==================== COLOR PALETTES ====================
COLOR_PALETTES = {
    'Mondrian': ['#FF0000', '#FFFF00', '#0000FF', '#FFFFFF', '#000000'],
    'Modern': ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2'],
    'Grayscale': ['#333333', '#666666', '#999999', '#CCCCCC'],
    'Vibrant': ['#E63946', '#F4A261', '#2A9D8F', '#E9C46A', '#264653'],
    'Pastel': ['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA', '#FFB3BA'],
    'Earth': ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F5DEB3'],
    'Cool': ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#023E8A']
}

# ==================== KD-TREE WITH WALL GAPS ====================
class KDNode3D:
    """Enhanced KD Node with wall gap awareness"""
    def __init__(self, x, y, z, w, d, h, depth, axis, is_wall_gap=False):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.depth = depth
        self.axis = axis
        self.children = []
        self.is_leaf = True
        self.color = None
        self.opacity = 0.8
        self.is_wall_gap = is_wall_gap  # Marks gap/void spaces

def generate_3d_kdtree_with_gaps(x, y, z, w, d, h, depth, max_depth, axis, 
                                randomness, wall_gap, rng):
    """Generate kd-tree with configurable wall gaps between rooms"""
    node = KDNode3D(x, y, z, w, d, h, depth, axis)
    
    # Half gap for symmetric distribution
    half_gap = wall_gap / 2.0
    
    # Stop recursion if depth reached or room too small
    min_room_size = 15 + wall_gap  # Ensure room big enough after gap
    if depth >= max_depth or min(w, d, h) < min_room_size:
        return node
    
    split_ratio = 0.3 + rng.random() * (0.7 - 0.3) * (1 - randomness)
    
    if axis == 0:  # Split along X axis (YZ plane)
        split_x = x + w * split_ratio
        
        # Check minimum room sizes with gaps
        if (split_x - x - half_gap > 10) and (x + w - split_x - half_gap > 10):
            # Left room
            node.children.append(generate_3d_kdtree_with_gaps(
                x, y, z, 
                split_x - x - half_gap, d, h,
                depth + 1, max_depth, 1, randomness, wall_gap, rng))
            
            # Right room
            node.children.append(generate_3d_kdtree_with_gaps(
                split_x + half_gap, y, z,
                x + w - split_x - half_gap, d, h,
                depth + 1, max_depth, 1, randomness, wall_gap, rng))
            
            node.is_leaf = False
            
    elif axis == 1:  # Split along Y axis (XZ plane)
        split_y = y + d * split_ratio
        
        if (split_y - y - half_gap > 10) and (y + d - split_y - half_gap > 10):
            # Back room
            node.children.append(generate_3d_kdtree_with_gaps(
                x, y, z,
                w, split_y - y - half_gap, h,
                depth + 1, max_depth, 2, randomness, wall_gap, rng))
            
            # Front room
            node.children.append(generate_3d_kdtree_with_gaps(
                x, split_y + half_gap, z,
                w, y + d - split_y - half_gap, h,
                depth + 1, max_depth, 2, randomness, wall_gap, rng))
            
            node.is_leaf = False
            
    else:  # axis == 2: Split along Z axis (XY plane)
        split_z = z + h * split_ratio
        
        if (split_z - z - half_gap > 10) and (z + h - split_z - half_gap > 10):
            # Bottom room
            node.children.append(generate_3d_kdtree_with_gaps(
                x, y, z,
                w, d, split_z - z - half_gap,
                depth + 1, max_depth, 0, randomness, wall_gap, rng))
            
            # Top room
            node.children.append(generate_3d_kdtree_with_gaps(
                x, y, split_z + half_gap,
                w, d, z + h - split_z - half_gap,
                depth + 1, max_depth, 0, randomness, wall_gap, rng))
            
            node.is_leaf = False
    
    return node

def assign_colors_3d(node, density, balance, palette, rng):
    """Assign Mondrian-style colors to leaf nodes"""
    if node.is_leaf:
        volume = node.w * node.d * node.h
        color_prob = density * (1 + balance * (volume / (200**3)))
        if rng.random() < min(color_prob, 0.9):
            node.color = rng.choice(palette)
            node.opacity = 0.9
        else:
            node.color = '#FFFFFF' if '#FFFFFF' in palette else '#F5F5F5'
            node.opacity = 0.5
    else:
        for child in node.children:
            assign_colors_3d(child, density, balance, palette, rng)

def add_cuboid_walls(fig, x, y, z, w, d, h, color, opacity, line_width=2):
    """Enhanced cuboid rendering with distinct wall edges"""
    vertices = np.array([
        [x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z],
        [x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h]
    ])
    
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    
    for face in faces:
        face_vertices = vertices[face]
        fig.add_trace(go.Mesh3d(
            x=face_vertices[:, 0], y=face_vertices[:, 1], z=face_vertices[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=opacity,
            showlegend=False, hoverinfo='none',
            lighting=dict(ambient=0.6, diffuse=0.5, specular=0.2)
        ))
    
    if line_width > 0:
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        edge_x, edge_y, edge_z = [], [], []
        for start, end in edges:
            edge_x.extend([vertices[start][0], vertices[end][0], None])
            edge_y.extend([vertices[start][1], vertices[end][1], None])
            edge_z.extend([vertices[start][2], vertices[end][2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode='lines',
            line=dict(color='#333333', width=line_width),
            showlegend=False, hoverinfo='none'
        ))

def render_3d_rooms(node, fig, show_walls=True):
    """Render rooms with optional wall visualization"""
    if node.is_leaf:
        add_cuboid_walls(fig, node.x, node.y, node.z, node.w, node.d, node.h, 
                        node.color, node.opacity, 2 if show_walls else 0)
    else:
        for child in node.children:
            render_3d_rooms(child, fig, show_walls)

def count_rooms(node):
    """Count total leaf rooms in tree"""
    if not node: return 0
    if node.is_leaf: return 1
    return count_rooms(node.children[0]) + count_rooms(node.children[1])

def collect_colored_rooms(node):
    """Generator for collecting colored rooms"""
    if node.is_leaf and node.color not in ['#FFFFFF', '#F5F5F5']:
        yield node
    elif not node.is_leaf:
        for child in node.children:
            yield from collect_colored_rooms(child)

def print_room_layout(node, level=0):
    """Print room hierarchy with dimensions and colors"""
    indent = "  " * level
    if node.is_leaf:
        color_info = node.color if node.color else "transparent"
        volume = node.w * node.d * node.h
        return f"{indent}🏠 Room: {node.w:.1f}×{node.d:.1f}×{node.h:.1f} (V:{volume:.0f}) - {color_info}"
    axis_name = ["X (East-West)", "Y (North-South)", "Z (Floor-Ceiling)"][node.axis]
    result = f"{indent}🧱 Wall split along {axis_name} (depth {node.depth})\n"
    result += print_room_layout(node.children[0], level + 1) + "\n"
    result += print_room_layout(node.children[1], level + 1)
    return result

def export_rooms_to_obj(node, wall_gap=0.0):
    """Export room layout to OBJ with gap consideration"""
    vertices = []
    faces = []
    colors = []
    
    def collect_mesh(node):
        nonlocal vertices, faces, colors
        if node.is_leaf:
            base_idx = len(vertices)
            v = [
                [node.x, node.y, node.z],
                [node.x + node.w, node.y, node.z],
                [node.x + node.w, node.y + node.d, node.z],
                [node.x, node.y + node.d, node.z],
                [node.x, node.y, node.z + node.h],
                [node.x + node.w, node.y, node.z + node.h],
                [node.x + node.w, node.y + node.d, node.z + node.h],
                [node.x, node.y + node.d, node.z + node.h]
            ]
            vertices.extend(v)
            colors.extend([node.color or '#EEEEEE'] * 8)
            
            face_indices = [
                [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],
                [base_idx+4, base_idx+5, base_idx+6], [base_idx+4, base_idx+6, base_idx+7],
                [base_idx, base_idx+1, base_idx+5], [base_idx, base_idx+5, base_idx+4],
                [base_idx+2, base_idx+3, base_idx+7], [base_idx+2, base_idx+7, base_idx+6],
                [base_idx, base_idx+3, base_idx+7], [base_idx, base_idx+7, base_idx+4],
                [base_idx+1, base_idx+2, base_idx+6], [base_idx+1, base_idx+6, base_idx+5]
            ]
            faces.extend(face_indices)
        else:
            for child in node.children:
                collect_mesh(child)
    
    collect_mesh(node)
    
    obj_content = f"# 3D Room Planner with Wall Gaps ({wall_gap} units)\n"
    obj_content += f"o RoomLayout\n\n"
    
    for i, v in enumerate(vertices):
        obj_content += f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n"
    
    obj_content += "\n"
    for face in faces:
        obj_content += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    
    return obj_content

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Room Planning Parameters")
    
    # Canvas and recursion settings
    col1, col2 = st.columns(2)
    with col1:
        canvas_size = st.slider("Building Size (px)", 100, 400, 250, 10)
        max_depth = st.slider("Max Floor Levels", 1, 5, 3)
    with col2:
        randomness = st.slider("Layout Randomness", 0.0, 1.0, 0.3, 0.05)
        wall_gap = st.slider("Wall Thickness (px)", 0.0, 15.0, 2.0, 0.5, 
                            help="Physical gap between rooms - simulates real wall thickness")
    
    st.divider()
    
    # Color settings
    st.subheader("🎨 Room Colors")
    col3, col4 = st.columns(2)
    with col3:
        color_density = st.slider("Color Fill %", 0.0, 1.0, 0.4, 0.05)
        color_balance = st.slider("Size Color Bias", 0.0, 1.0, 0.6)
    with col4:
        palette_name = st.selectbox("Color Palette", list(COLOR_PALETTES.keys()))
        show_wireframe = st.checkbox("Show Room Edges", True)
    
    st.divider()
    
    # Random seed and generation
    seed = st.number_input("Random Seed", 0, 9999, 42)
    generate_clicked = st.button("🏗️ Generate Room Layout", type="primary", width='stretch')
    
    # Wall gap info
    if wall_gap > 0:
        st.info(f"💡 Wall gap of {wall_gap:.1f}px creates physical separation between rooms")

# ==================== INITIALIZE SESSION STATE ====================
if 'root' not in st.session_state:
    st.session_state.root = None
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'wall_gap' not in st.session_state:
    st.session_state.wall_gap = 0.0

# ==================== GENERATE ROOM LAYOUT ====================
if generate_clicked or not st.session_state.generated:
    if seed == 0:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    
    class CustomRNG:
        def __init__(self, rng): self.rng = rng
        def random(self): return self.rng.rand()
        def choice(self, arr): return arr[self.rng.randint(len(arr))]
    
    custom_rng = CustomRNG(rng)
    palette = COLOR_PALETTES[palette_name]
    
    margin = 20
    root = generate_3d_kdtree_with_gaps(
        margin, margin, margin, 
        canvas_size - 2*margin, 
        canvas_size - 2*margin,
        canvas_size - 2*margin,
        0, max_depth, 0, randomness, wall_gap, custom_rng)
    
    assign_colors_3d(root, color_density, color_balance, palette, custom_rng)
    
    st.session_state.root = root
    st.session_state.generated = True
    st.session_state.canvas_size = canvas_size
    st.session_state.max_depth = max_depth
    st.session_state.show_wireframe = show_wireframe
    st.session_state.wall_gap = wall_gap

# ==================== RENDER ROOM LAYOUT ====================
if st.session_state.root:
    fig = go.Figure()
    render_3d_rooms(st.session_state.root, fig, st.session_state.show_wireframe)
    
    total_rooms = count_rooms(st.session_state.root)
    colored_rooms = sum(1 for _ in collect_colored_rooms(st.session_state.root))
    
    fig.update_layout(
        title=f"Room Layout | {total_rooms} Rooms | Wall Gap: {st.session_state.wall_gap:.1f}px",
        scene=dict(
            xaxis_title="X (Width)", yaxis_title="Y (Depth)", zaxis_title="Z (Height)",
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.5)),
            bgcolor='#f3f2ee'
        ),
        paper_bgcolor='#f3f2ee',
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        uirevision='constant'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== STATISTICS ====================
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("🏠 Rooms", total_rooms)
    with col2:
        st.metric("🧱 Wall Splits", st.session_state.max_depth)
    with col3:
        st.metric("📦 Building Size", f"{st.session_state.canvas_size}³")
    with col4:
        st.metric("🎨 Colored Rooms", colored_rooms)
    with col5:
        ratio = colored_rooms / total_rooms if total_rooms > 0 else 0
        st.metric("🎯 Color Density", f"{ratio:.0%}")
    with col6:
        st.metric("📏 Wall Gap", f"{st.session_state.wall_gap:.1f}px")
    
    # ==================== EXPORT OPTIONS ====================
    st.divider()
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    
    with col_export1:
        st.download_button(
            label="📦 Export Rooms (OBJ)",
            data=export_rooms_to_obj(st.session_state.root, st.session_state.wall_gap),
            file_name="room_layout.obj",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export2:
        st.download_button(
            label="📊 Room Layout (TXT)",
            data=print_room_layout(st.session_state.root),
            file_name="room_layout.txt",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export3:
        # Export room dimensions for architectural use
        room_data = f"Room Layout with {st.session_state.wall_gap:.1f}px wall gaps\n"
        room_data += f"Total rooms: {total_rooms}\n"
        room_data += f"Building dimensions: {st.session_state.canvas_size}³\n\n"
        room_data += print_room_layout(st.session_state.root)
        
        st.download_button(
            label="🏗️ Architectural Specs",
            data=room_data,
            file_name="room_specifications.txt",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export4:
        if st.button("📸 View Guide", width='stretch'):
            st.info("Use browser screenshot or right-click on 3D view to save image")
    
    # ==================== ROOM LAYOUT VIEWER ====================
    with st.expander("🏗️ View Room Hierarchy"):
        st.markdown(f"**Wall Gap:** {st.session_state.wall_gap:.1f}px | **Max Depth:** {st.session_state.max_depth}")
        st.code(print_room_layout(st.session_state.root), language="text")

# ==================== WALL GAP VISUALIZATION ====================
if st.session_state.root and st.session_state.wall_gap > 0:
    with st.expander("📏 Wall Gap Analysis"):
        total_volume = st.session_state.canvas_size ** 3
        room_volume = sum(node.w * node.d * node.h 
                         for node in [st.session_state.root] 
                         if hasattr(node, 'w'))
        
        # Calculate approximate wall volume
        wall_volume = total_volume - room_volume
        wall_percentage = (wall_volume / total_volume * 100) if total_volume > 0 else 0
        
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.metric("Room Volume", f"{room_volume:.0f} px³")
        with col_w2:
            st.metric("Wall Volume", f"{wall_volume:.0f} px³")
        with col_w3:
            st.metric("Wall %", f"{wall_percentage:.1f}%")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏠 <strong>3D Room Planner</strong> | Mondrian-inspired recursive partitioning with real wall gaps</p>
    <p><em>Exportable models for architectural planning and interior design</em></p>
</div>
""", unsafe_allow_html=True)
