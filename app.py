import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import io

st.set_page_config(
    page_title="3D Mondrian Floor Planner - Controlled Floors",
    page_icon="🏗️",
    layout="wide"
)

# ==================== SOCIAL SHARING ====================
st.markdown("""
    <meta property="og:title" content="3D Mondrian Floor Planner - Custom Floor Count">
    <meta property="og:description" content="3D kd-tree recursion for architectural floor planning with precise floor control">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://mondrian-planner.streamlit.app">
    <meta property="og:image" content="https://raw.githubusercontent.com/yusdesign/mondrian-planner/main/og_image.png">
    <meta name="twitter:card" content="summary_large_image">
""", unsafe_allow_html=True)

st.title("🏗️ 3D Mondrian Floor Planner")
st.markdown("### *kd-Tree Recursion for Architectural Floor Planning*")

# ==================== COLOR PALETTES ====================
COLOR_PALETTES = {
    'Mondrian': ['#FF0000', '#FFFF00', '#0000FF', '#FFFFFF', '#000000'],
    'Modern': ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2'],
    'Architectural': ['#C0C0C0', '#D3D3D3', '#A9A9A9', '#808080', '#696969'],
    'Vibrant': ['#E63946', '#F4A261', '#2A9D8F', '#E9C46A', '#264653'],
    'Pastel': ['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA', '#FFB3BA'],
    'Earth': ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F5DEB3'],
    'Cool': ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#023E8A']
}

# ==================== CLASSICAL KD-TREE FLOOR NODE ====================
class FloorNode:
    """
    Classical 3D Mondrian kd-Tree Node
    Floor = Horizontal slice created by Z-axis split
    Room = Horizontal subdivision within a floor via X/Y splits
    """
    def __init__(self, x, y, z, w, d, h, depth, axis, floor_level=0, is_floor_split=False):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.depth = depth
        self.axis = axis  # 0=X, 1=Y, 2=Z (floor split)
        self.children = []
        self.is_leaf = True
        self.color = None
        self.opacity = 0.8
        self.floor_level = floor_level  # Floor index (0-based)
        self.is_floor_split = is_floor_split  # True if this split creates floors

def generate_mondrian_floors(x, y, z, w, d, h, depth, max_depth,
                             target_floors, randomness, wall_gap,
                             floor_level, rng):
    """
    Classical Mondrian Floor Generation:
    1. First, recursively split Z-axis to create target_floors
    2. Then, subdivide each floor with alternating X/Y splits
    
    Floor definition: All rooms with same floor_level share Z-range
    """
    node = FloorNode(x, y, z, w, d, h, depth, 0, floor_level)
    
    half_gap = wall_gap / 2.0
    min_room_size = 15 + wall_gap
    
    if depth >= max_depth or min(w, d, h) < min_room_size:
        return node
    
    # Determine what this depth level should do
    floors_created_so_far = 2 ** (floor_level.bit_length())  # ≈ 2^ceil(log2(floor_level+1))
    need_more_floors = floor_level + 1 < target_floors
    
    # Strategy: Create floors first, then subdivide rooms
    if need_more_floors and depth < target_floors - 1:
        # Floor creation phase: Force Z-axis splits
        axis = 2
        node.is_floor_split = True
    else:
        # Room subdivision phase: Alternate X/Y within floors
        axis = depth % 2  # X or Y only
    
    node.axis = axis
    split_ratio = 0.3 + rng.random() * (0.7 - 0.3) * (1 - randomness)
    
    if axis == 0:  # X-split: Room width division (same floor)
        split_x = x + w * split_ratio
        if (split_x - x - half_gap > 10) and (x + w - split_x - half_gap > 10):
            node.children.append(generate_mondrian_floors(
                x, y, z, split_x - x - half_gap, d, h,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level, rng))
            node.children.append(generate_mondrian_floors(
                split_x + half_gap, y, z, x + w - split_x - half_gap, d, h,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level, rng))
            node.is_leaf = False
            
    elif axis == 1:  # Y-split: Room depth division (same floor)
        split_y = y + d * split_ratio
        if (split_y - y - half_gap > 10) and (y + d - split_y - half_gap > 10):
            node.children.append(generate_mondrian_floors(
                x, y, z, w, split_y - y - half_gap, h,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level, rng))
            node.children.append(generate_mondrian_floors(
                x, split_y + half_gap, z, w, y + d - split_y - half_gap, h,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level, rng))
            node.is_leaf = False
            
    else:  # axis == 2: Z-split = NEW FLOOR CREATED
        split_z = z + h * split_ratio
        if (split_z - z - half_gap > 10) and (z + h - split_z - half_gap > 10):
            # LOWER FLOOR (same floor_level)
            node.children.append(generate_mondrian_floors(
                x, y, z, w, d, split_z - z - half_gap,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level, rng))
            # UPPER FLOOR (increment floor_level)
            node.children.append(generate_mondrian_floors(
                x, y, split_z + half_gap, w, d, z + h - split_z - half_gap,
                depth + 1, max_depth, target_floors, randomness, wall_gap,
                floor_level + 1, rng))
            node.is_leaf = False
    
    return node

def assign_mondrian_colors(node, density, balance, palette, rng):
    """Assign Mondrian-style colors with floor-based variations"""
    if node.is_leaf:
        volume = node.w * node.d * node.h
        color_prob = density * (1 + balance * (volume / (200**3)))
        
        if rng.random() < min(color_prob, 0.9):
            node.color = rng.choice(palette)
            node.opacity = 0.9
        else:
            node.color = '#FFFFFF' if '#FFFFFF' in palette else '#F5F5F5'
            node.opacity = 0.3
    else:
        for child in node.children:
            assign_mondrian_colors(child, density, balance, palette, rng)

def add_cuboid_to_scene(fig, x, y, z, w, d, h, color, opacity, floor_level, line_width=2):
    """Add 3D room cuboid with floor metadata"""
    vertices = np.array([
        [x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z],
        [x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h]
    ])
    
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    
    # Floor-based visual depth
    floor_shade = 1.0 - (floor_level * 0.08)
    adjusted_opacity = max(0.2, opacity * floor_shade)
    
    for face in faces:
        face_vertices = vertices[face]
        fig.add_trace(go.Mesh3d(
            x=face_vertices[:, 0], y=face_vertices[:, 1], z=face_vertices[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=adjusted_opacity,
            showlegend=False,
            hoverinfo='text',
            hovertext=f'<b>Floor {floor_level + 1}</b><br>Room: {w:.0f}×{d:.0f}×{h:.0f}',
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
            showlegend=False,
            hoverinfo='skip'
        ))

def render_mondrian_scene(node, fig, show_walls=True):
    """Render complete Mondrian floor plan"""
    if node.is_leaf:
        add_cuboid_to_scene(fig, node.x, node.y, node.z, node.w, node.d, node.h,
                           node.color, node.opacity, node.floor_level,
                           2 if show_walls else 0)
    else:
        for child in node.children:
            render_mondrian_scene(child, fig, show_walls)

def count_floors_and_rooms(node):
    """Count distinct floors and rooms per floor"""
    floor_data = defaultdict(lambda: {'rooms': 0, 'total_volume': 0})
    
    def traverse(n):
        if n.is_leaf:
            floor_data[n.floor_level]['rooms'] += 1
            floor_data[n.floor_level]['total_volume'] += n.w * n.d * n.h
        else:
            for child in n.children:
                traverse(child)
    
    traverse(node)
    
    return {
        'total_floors': len(floor_data),
        'floor_data': dict(sorted(floor_data.items())),
        'total_rooms': sum(f['rooms'] for f in floor_data.values())
    }

def print_mondrian_structure(node, level=0):
    """Print classical Mondrian floor structure"""
    indent = "  " * level
    if node.is_leaf:
        return (f"{indent}🏠 Floor {node.floor_level + 1} Room: "
                f"{node.w:.1f}×{node.d:.1f}×{node.h:.1f} "
                f"[{node.color}]")
    
    split_type = "🏢 FLOOR SPLIT (Z-axis)" if node.is_floor_split else "🧱 Room Division"
    axis_name = {0: "X (Width)", 1: "Y (Depth)", 2: "Z (Floor Level)"}[node.axis]
    result = f"{indent}🔪 {split_type} along {axis_name} (depth {node.depth})\n"
    result += print_mondrian_structure(node.children[0], level + 1) + "\n"
    result += print_mondrian_structure(node.children[1], level + 1)
    return result

def export_mondrian_obj(node, wall_gap=0.0):
    """Export Mondrian floor plan to OBJ format"""
    vertices = []
    faces = []
    floor_groups = defaultdict(list)
    
    def collect_mesh(n):
        nonlocal vertices, faces
        if n.is_leaf:
            base_idx = len(vertices)
            v = [
                [n.x, n.y, n.z], [n.x + n.w, n.y, n.z],
                [n.x + n.w, n.y + n.d, n.z], [n.x, n.y + n.d, n.z],
                [n.x, n.y, n.z + n.h], [n.x + n.w, n.y, n.z + n.h],
                [n.x + n.w, n.y + n.d, n.z + n.h], [n.x, n.y + n.d, n.z + n.h]
            ]
            vertices.extend(v)
            
            face_indices = [
                [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],
                [base_idx+4, base_idx+5, base_idx+6], [base_idx+4, base_idx+6, base_idx+7],
                [base_idx, base_idx+1, base_idx+5], [base_idx, base_idx+5, base_idx+4],
                [base_idx+2, base_idx+3, base_idx+7], [base_idx+2, base_idx+7, base_idx+6],
                [base_idx, base_idx+3, base_idx+7], [base_idx, base_idx+7, base_idx+4],
                [base_idx+1, base_idx+2, base_idx+6], [base_idx+1, base_idx+6, base_idx+5]
            ]
            floor_groups[n.floor_level].extend(face_indices)
            faces.extend(face_indices)
        else:
            for child in n.children:
                collect_mesh(child)
    
    collect_mesh(node)
    
    obj = f"# Mondrian Floor Plan - {len(floor_groups)} Floors, Wall Gap: {wall_gap}px\n"
    for floor_num in sorted(floor_groups.keys()):
        obj += f"o Floor_{floor_num + 1}\n"
    
    obj += "\n"
    for v in vertices:
        obj += f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n"
    
    obj += "\n"
    for face in faces:
        obj += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    
    return obj

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Floor Planning Controls")
    
    # Primary floor control
    st.subheader("🏢 Floor Configuration")
    
    num_floors = st.radio(
        "Number of Floors",
        options=[1, 2, 3],
        index=1,
        horizontal=True,
        help="1=Single story, 2=Ground+Upper, 3=Ground+Middle+Upper"
    )
    
    # Depth adjusts room subdivision per floor
    max_depth = st.slider(
        "Room Subdivision Depth",
        1, 5, 3,
        help="Controls how many times rooms are subdivided. Higher = smaller rooms"
    )
    
    st.divider()
    
    # Building parameters
    st.subheader("📐 Building Parameters")
    col1, col2 = st.columns(2)
    with col1:
        canvas_size = st.slider("Building Size", 100, 400, 250, 10)
        randomness = st.slider("Layout Randomness", 0.0, 1.0, 0.3, 0.05)
    with col2:
        wall_gap = st.slider("Wall Thickness", 0.0, 15.0, 2.0, 0.5,
                            help="Physical separation between rooms and floors")
    
    st.divider()
    
    # Color settings
    st.subheader("🎨 Mondrian Colors")
    col3, col4 = st.columns(2)
    with col3:
        color_density = st.slider("Color Density", 0.0, 1.0, 0.4, 0.05)
        color_balance = st.slider("Volume Balance", 0.0, 1.0, 0.6)
    with col4:
        palette_name = st.selectbox("Palette", list(COLOR_PALETTES.keys()))
        show_wireframe = st.checkbox("Show Edges", True)
    
    st.divider()
    
    seed = st.number_input("Random Seed", 0, 9999, 42)
    
    generate_clicked = st.button(
        "🏗️ Generate Floor Plan",
        type="primary",
        width='stretch'
    )
    
    # Info display
    st.caption(f"📏 Floor definition: Z-axis splits create floors, X/Y splits create rooms within floors")
    if wall_gap > 0:
        st.success(f"🧱 Wall gaps: {wall_gap:.1f}px between rooms and floors")

# ==================== SESSION STATE ====================
if 'root' not in st.session_state:
    st.session_state.root = None
if 'generated' not in st.session_state:
    st.session_state.generated = False

# ==================== GENERATE FLOOR PLAN ====================
if generate_clicked or not st.session_state.generated:
    if seed == 0:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    
    class CustomRNG:
        def __init__(self, rng):
            self.rng = rng
        def random(self):
            return self.rng.rand()
        def choice(self, arr):
            return arr[self.rng.randint(len(arr))]
        def randint(self, low, high=None):
            if high is None:
                high = low
                low = 0
            return self.rng.randint(low, high)
    
    custom_rng = CustomRNG(rng)
    palette = COLOR_PALETTES[palette_name]
    
    margin = 20
    
    # Generate classical Mondrian floor plan
    root = generate_mondrian_floors(
        margin, margin, margin,
        canvas_size - 2*margin,
        canvas_size - 2*margin,
        canvas_size - 2*margin,
        0, max_depth, num_floors, randomness, wall_gap,
        0, custom_rng
    )
    
    assign_mondrian_colors(root, color_density, color_balance, palette, custom_rng)
    
    st.session_state.root = root
    st.session_state.generated = True
    st.session_state.canvas_size = canvas_size
    st.session_state.num_floors = num_floors
    st.session_state.max_depth = max_depth
    st.session_state.show_wireframe = show_wireframe
    st.session_state.wall_gap = wall_gap

# ==================== RENDER & DISPLAY ====================
if st.session_state.root:
    # 3D Visualization
    fig = go.Figure()
    render_mondrian_scene(st.session_state.root, fig, st.session_state.show_wireframe)
    
    stats = count_floors_and_rooms(st.session_state.root)
    
    fig.update_layout(
        title=dict(
            text=f"🏗️ Mondrian Floor Plan | {stats['total_floors']} Floors | {stats['total_rooms']} Rooms",
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title="X (Width)", yaxis_title="Y (Depth)", zaxis_title="Z (Floor Height)",
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor='#f3f2ee'
        ),
        paper_bgcolor='#f3f2ee',
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        uirevision='constant'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Floor Statistics
    st.subheader("📊 Floor Statistics")
    
    cols = st.columns(min(stats['total_floors'], 3))
    for floor_num, (floor_idx, data) in enumerate(sorted(stats['floor_data'].items())):
        with cols[floor_num]:
            avg_room_vol = data['total_volume'] / data['rooms'] if data['rooms'] > 0 else 0
            st.metric(
                f"Floor {floor_idx + 1}",
                f"{data['rooms']} rooms",
                f"Avg: {avg_room_vol:.0f}px³"
            )
    
    # Export Options
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📦 Export OBJ",
            export_mondrian_obj(st.session_state.root, st.session_state.wall_gap),
            "mondrian_floors.obj",
            "text/plain",
            width='stretch'
        )
    
    with col2:
        st.download_button(
            "📊 Floor Structure",
            print_mondrian_structure(st.session_state.root),
            "mondrian_structure.txt",
            "text/plain",
            width='stretch'
        )
    
    with col3:
        # Detailed floor report
        report = "MONDRIAN FLOOR PLAN REPORT\n"
        report += f"Floors: {stats['total_floors']}\n"
        report += f"Total Rooms: {stats['total_rooms']}\n"
        report += f"Wall Gap: {st.session_state.wall_gap}px\n\n"
        
        for floor_idx, data in sorted(stats['floor_data'].items()):
            report += f"Floor {floor_idx + 1}: {data['rooms']} rooms\n"
            report += f"  Total Volume: {data['total_volume']:.0f}px³\n"
            report += f"  Avg Room Size: {data['total_volume']/data['rooms']:.0f}px³\n\n"
        
        st.download_button(
            "📋 Floor Report",
            report,
            "floor_report.txt",
            "text/plain",
            width='stretch'
        )
    
    # Floor Structure Viewer
    with st.expander("🏗️ Mondrian Floor Structure"):
        st.code(print_mondrian_structure(st.session_state.root), language="text")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏗️ <strong>3D Mondrian kd-Tree Floor Planner</strong></p>
    <p><em>Classical floor definition: Z-axis = Floor levels, X/Y = Room subdivisions</em></p>
    <p style="font-size: 0.8em;">Floor: Horizontal spatial slice | Room: Subdivided space within a floor</p>
</div>
""", unsafe_allow_html=True)
