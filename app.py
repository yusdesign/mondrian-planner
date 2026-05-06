import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import io
import base64
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="3D Mondrian Room Planner - Vertical Floor Stacking",
    page_icon="🏢",
    layout="wide"
)

# ==================== SOCIAL SHARING ====================
st.markdown("""
    <meta property="og:title" content="3D Vertical Floor Stacking Room Planner">
    <meta property="og:description" content="Interactive 3D room partitioning with proper vertical floor levels">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://mondrian-planner.streamlit.app">
    <meta property="og:image" content="https://raw.githubusercontent.com/yusdesign/mondrian-planner/main/og_image.png">
    <meta name="twitter:card" content="summary_large_image">
""", unsafe_allow_html=True)

st.title("🏢 3D Vertical Floor Room Planner")
st.markdown("### *Recursive axis-aligned partitioning with proper floor stacking and wall gaps*")

# ==================== COLOR PALETTES ====================
COLOR_PALETTES = {
    'Mondrian': ['#FF0000', '#FFFF00', '#0000FF', '#FFFFFF', '#000000'],
    'Modern': ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2'],
    'Grayscale': ['#333333', '#666666', '#999999', '#CCCCCC'],
    'Vibrant': ['#E63946', '#F4A261', '#2A9D8F', '#E9C46A', '#264653'],
    'Pastel': ['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA', '#FFB3BA'],
    'Earth': ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F5DEB3'],
    'Cool': ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#023E8A'],
    'Architectural': ['#C0C0C0', '#D3D3D3', '#A9A9A9', '#808080', '#696969']
}

# ==================== ENHANCED KD-TREE FOR VERTICAL FLOORS ====================
class FloorNode:
    """Enhanced KD Node with floor-aware splitting"""
    def __init__(self, x, y, z, w, d, h, depth, axis, floor_level=0):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.depth = depth
        self.axis = axis
        self.children = []
        self.is_leaf = True
        self.color = None
        self.opacity = 0.8
        self.floor_level = floor_level  # Track which floor this room is on
        self.is_vertical_split = False  # Track if this was a floor-to-floor split

def generate_floor_based_kdtree(x, y, z, w, d, h, depth, max_depth, 
                               randomness, wall_gap, floor_level, rng):
    """
    Generate kd-tree with proper vertical floor stacking first,
    then horizontal room subdivision on each floor
    """
    node = FloorNode(x, y, z, w, d, h, depth, 0, floor_level)
    
    half_gap = wall_gap / 2.0
    min_room_size = 15 + wall_gap
    
    if depth >= max_depth or min(w, d, h) < min_room_size:
        return node
    
    # Determine split axis based on floor-stacking strategy
    # Floors get created first (Z-axis splits), then rooms within floors (X/Y splits)
    floors_desired = max(1, max_depth // 2 + 1)
    
    if depth < floors_desired and depth % 2 == 0:
        # Floor split: Divide vertically (Z-axis) to create floors
        axis = 2
        node.is_vertical_split = True
    elif depth < floors_desired:
        # On-floor subdivision: Divide horizontally for rooms on current floor
        axis = depth % 2  # Alternate X(0) and Y(1)
    else:
        # Further subdivision within rooms
        axis = depth % 2  # Continue alternating X and Y
    
    node.axis = axis
    split_ratio = 0.3 + rng.random() * (0.7 - 0.3) * (1 - randomness)
    
    if axis == 0:  # Split along X (Width division on same floor)
        split_x = x + w * split_ratio
        if (split_x - x - half_gap > 10) and (x + w - split_x - half_gap > 10):
            node.children.append(generate_floor_based_kdtree(
                x, y, z, split_x - x - half_gap, d, h,
                depth + 1, max_depth, randomness, wall_gap, floor_level, rng))
            node.children.append(generate_floor_based_kdtree(
                split_x + half_gap, y, z, x + w - split_x - half_gap, d, h,
                depth + 1, max_depth, randomness, wall_gap, floor_level, rng))
            node.is_leaf = False
            
    elif axis == 1:  # Split along Y (Depth division on same floor)
        split_y = y + d * split_ratio
        if (split_y - y - half_gap > 10) and (y + d - split_y - half_gap > 10):
            node.children.append(generate_floor_based_kdtree(
                x, y, z, w, split_y - y - half_gap, h,
                depth + 1, max_depth, randomness, wall_gap, floor_level, rng))
            node.children.append(generate_floor_based_kdtree(
                x, split_y + half_gap, z, w, y + d - split_y - half_gap, h,
                depth + 1, max_depth, randomness, wall_gap, floor_level, rng))
            node.is_leaf = False
            
    else:  # axis == 2: Split along Z (Floor stacking - different floor levels)
        split_z = z + h * split_ratio
        if (split_z - z - half_gap > 10) and (z + h - split_z - half_gap > 10):
            # Lower floor (FLOOR_LEVEL STAYS SAME)
            node.children.append(generate_floor_based_kdtree(
                x, y, z, w, d, split_z - z - half_gap,
                depth + 1, max_depth, randomness, wall_gap, floor_level, rng))
            # Upper floor (FLOOR_LEVEL INCREASES BY 1)
            node.children.append(generate_floor_based_kdtree(
                x, y, split_z + half_gap, w, d, z + h - split_z - half_gap,
                depth + 1, max_depth, randomness, wall_gap, floor_level + 1, rng))
            node.is_leaf = False
    
    return node

def assign_colors_by_floor(node, density, balance, palette, rng):
    """
    Assign colors with floor consideration - 
    each floor gets a color theme variation
    """
    if node.is_leaf:
        volume = node.w * node.d * node.h
        color_prob = density * (1 + balance * (volume / (200**3)))
        
        # Shift palette index based on floor level for visual floor distinction
        if rng.random() < min(color_prob, 0.9):
            # Use the rng.choice method from our CustomRNG class
            # But shift the palette based on floor level for visual clustering
            floor_offset = node.floor_level % len(palette)
            
            # Instead of randint, use rng.choice with all palette colors
            # and apply floor-based weighting
            if hasattr(rng, 'choice'):
                node.color = rng.choice(palette)
            else:
                # Fallback: simple random selection with floor influence
                import random
                node.color = palette[random.randint(0, len(palette)-1)]
            
            node.opacity = 0.9
        else:
            node.color = '#FFFFFF' if '#FFFFFF' in palette else '#F5F5F5'
            node.opacity = 0.3
    else:
        for child in node.children:
            assign_colors_by_floor(child, density, balance, palette, rng)

def add_room_cuboid(fig, x, y, z, w, d, h, color, opacity, floor_level, line_width=2):
    """Render room cuboid with floor-aware styling"""
    vertices = np.array([
        [x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z],
        [x, y, z + h], [x + w, y, z + h], [x + w, y + d, z + h], [x, y + d, z + h]
    ])
    
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
    ]
    
    # Darker shade for lower floors to show depth
    floor_darkness = 1.0 - (floor_level * 0.05)
    adjusted_opacity = opacity * floor_darkness
    
    for face in faces:
        face_vertices = vertices[face]
        fig.add_trace(go.Mesh3d(
            x=face_vertices[:, 0], y=face_vertices[:, 1], z=face_vertices[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=adjusted_opacity,
            showlegend=False, hoverinfo='text',
            hovertext=f'Floor {floor_level + 1}',
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
            showlegend=False, hoverinfo='text',
            hovertext=f'Floor {floor_level + 1} wall'
        ))

def render_floor_rooms(node, fig, show_walls=True):
    """Render all rooms organized by floors"""
    if node.is_leaf:
        add_room_cuboid(fig, node.x, node.y, node.z, node.w, node.d, node.h,
                       node.color, node.opacity, node.floor_level, 
                       2 if show_walls else 0)
    else:
        for child in node.children:
            render_floor_rooms(child, fig, show_walls)

def count_floors(node):
    """Count distinct floor levels in the layout"""
    floors = set()
    
    def traverse(n):
        if n.is_leaf:
            floors.add(n.floor_level)
        else:
            for child in n.children:
                traverse(child)
    
    traverse(node)
    return len(floors)

def count_rooms_per_floor(node):
    """Count rooms on each floor level"""
    floor_rooms = defaultdict(int)
    
    def traverse(n):
        if n.is_leaf:
            floor_rooms[n.floor_level] += 1
        else:
            for child in n.children:
                traverse(child)
    
    traverse(node)
    return dict(sorted(floor_rooms.items()))

def collect_rooms_by_floor(node):
    """Generator that yields rooms organized by floor"""
    rooms_by_floor = defaultdict(list)
    
    def traverse(n):
        if n.is_leaf:
            rooms_by_floor[n.floor_level].append(n)
        else:
            for child in n.children:
                traverse(child)
    
    traverse(node)
    return dict(sorted(rooms_by_floor.items()))

def print_floor_layout(node, level=0):
    """Print floor-by-floor room layout"""
    indent = "  " * level
    if node.is_leaf:
        color_info = node.color if node.color else "transparent"
        return (f"{indent}🏠 Floor {node.floor_level + 1} Room: {node.w:.1f}×{node.d:.1f}×{node.h:.1f} "
               f"(V:{node.w*node.d*node.h:.0f}) - {color_info}")
    
    split_type = "🏢 Floor Split (Z-axis)" if node.is_vertical_split else "🧱 Wall Split"
    axis_name = ["X (Width)", "Y (Depth)", "Z (Floor Level)"][node.axis]
    result = f"{indent}{split_type} along {axis_name} (depth {node.depth})\n"
    result += print_floor_layout(node.children[0], level + 1) + "\n"
    result += print_floor_layout(node.children[1], level + 1)
    return result

def export_floor_rooms_obj(node, wall_gap=0.0):
    """Export room layout with floor organization"""
    vertices = []
    faces = []
    floor_groups = defaultdict(list)
    
    def collect_mesh(node):
        nonlocal vertices, faces
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
            
            face_indices = [
                [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],
                [base_idx+4, base_idx+5, base_idx+6], [base_idx+4, base_idx+6, base_idx+7],
                [base_idx, base_idx+1, base_idx+5], [base_idx, base_idx+5, base_idx+4],
                [base_idx+2, base_idx+3, base_idx+7], [base_idx+2, base_idx+7, base_idx+6],
                [base_idx, base_idx+3, base_idx+7], [base_idx, base_idx+7, base_idx+4],
                [base_idx+1, base_idx+2, base_idx+6], [base_idx+1, base_idx+6, base_idx+5]
            ]
            floor_groups[node.floor_level].extend(face_indices)
            faces.extend(face_indices)
        else:
            for child in node.children:
                collect_mesh(child)
    
    collect_mesh(node)
    
    obj_content = f"# 3D Floor-based Room Layout with {wall_gap}px wall gaps\n"
    obj_content += f"# Total floors: {count_floors(node)}\n"
    
    for floor_num, floor_faces in sorted(floor_groups.items()):
        obj_content += f"o Floor_{floor_num + 1}\n"
    
    obj_content += "\n"
    for v in vertices:
        obj_content += f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n"
    
    obj_content += "\n"
    for face in faces:
        obj_content += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    
    return obj_content

# ==================== SIDEBAR CONTROLS ====================
with st.sidebar:
    st.header("⚙️ Floor Planning Parameters")
    
    # Building and floor settings
    col1, col2 = st.columns(2)
    with col1:
        canvas_size = st.slider("Building Size (px)", 100, 400, 250, 10)
        max_depth = st.slider("Max Subdivision Depth", 1, 5, 3,
                             help="Higher values = more room subdivision after floors")
    with col2:
        randomness = st.slider("Layout Randomness", 0.0, 1.0, 0.3, 0.05)
        wall_gap = st.slider("Wall Thickness (px)", 0.0, 15.0, 2.0, 0.5,
                            help="Physical gap between rooms and floors")
    
    # Floor calculation preview
    num_floors = max(1, max_depth // 2 + 1)
    st.info(f"📊 This will create approximately **{num_floors} floors** with rooms on each level")
    
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
    
    # Generation controls
    seed = st.number_input("Random Seed", 0, 9999, 42)
    generate_clicked = st.button("🏗️ Generate Floor Layout", type="primary", width='stretch')
    
    if wall_gap > 0:
        st.info(f"💡 {wall_gap:.1f}px wall gap creates real physical separation")

# ==================== INITIALIZE SESSION STATE ====================
if 'root' not in st.session_state:
    st.session_state.root = None
if 'generated' not in st.session_state:
    st.session_state.generated = False
if 'wall_gap' not in st.session_state:
    st.session_state.wall_gap = 0.0

# ==================== GENERATE FLOOR LAYOUT ====================
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
            """Added randint method to match numpy.random.RandomState interface"""
            if high is None:
                high = low
                low = 0
            return self.rng.randint(low, high)
    
    custom_rng = CustomRNG(rng)
    palette = COLOR_PALETTES[palette_name]
    
    margin = 20
    
    # Generate the floor-based kd-tree
    root = generate_floor_based_kdtree(
        margin, margin, margin,
        canvas_size - 2*margin,
        canvas_size - 2*margin,
        canvas_size - 2*margin,
        0, max_depth, randomness, wall_gap, 0, custom_rng)
    
    assign_colors_by_floor(root, color_density, color_balance, palette, custom_rng)
    
    st.session_state.root = root
    st.session_state.generated = True
    st.session_state.canvas_size = canvas_size
    st.session_state.max_depth = max_depth
    st.session_state.show_wireframe = show_wireframe
    st.session_state.wall_gap = wall_gap

# ==================== RENDER FLOOR LAYOUT ====================
if st.session_state.root:
    fig = go.Figure()
    render_floor_rooms(st.session_state.root, fig, st.session_state.show_wireframe)
    
    total_floors = count_floors(st.session_state.root)
    total_rooms = count_rooms_per_floor(st.session_state.root)
    total_room_count = sum(total_rooms.values())
    
    title_text = (
        f"Floor Layout | {total_floors} Floors | "
        f"{total_room_count} Rooms | Wall Gap: {st.session_state.wall_gap:.1f}px"
    )
    
    fig.update_layout(
        title=title_text,
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
    
    # ==================== FLOOR STATISTICS ====================
    st.subheader("📊 Floor-by-Floor Statistics")
    
    rooms_by_floor = collect_rooms_by_floor(st.session_state.root)
    
    # Create columns for each floor
    if total_floors > 0:
        cols = st.columns(min(total_floors, 5))
        for floor_num in range(total_floors):
            col_idx = floor_num % 5
            with cols[col_idx]:
                rooms_on_floor = rooms_by_floor.get(floor_num, [])
                floor_room_count = len(rooms_on_floor)
                floor_volume = sum(r.w * r.d * r.h for r in rooms_on_floor)
                
                st.metric(
                    f"Floor {floor_num + 1}",
                    f"{floor_room_count} rooms",
                    f"V:{floor_volume:.0f}px³"
                )
    
    # ==================== EXPORT OPTIONS ====================
    st.divider()
    col_export1, col_export2, col_export3, col_export4 = st.columns(4)
    
    with col_export1:
        st.download_button(
            label="📦 Export Rooms (OBJ)",
            data=export_floor_rooms_obj(st.session_state.root, st.session_state.wall_gap),
            file_name="floor_layout.obj",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export2:
        st.download_button(
            label="📊 Floor Layout (TXT)",
            data=print_floor_layout(st.session_state.root),
            file_name="floor_layout.txt",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export3:
        # Export detailed floor-by-floor specs
        if rooms_by_floor:
            floor_specs = f"FLOOR-BY-FLOOR ROOM LAYOUT\n"
            floor_specs += f"Wall Gap: {st.session_state.wall_gap:.1f}px\n"
            floor_specs += f"Total Floors: {total_floors}\n\n"
            
            for floor_num, rooms in sorted(rooms_by_floor.items()):
                floor_specs += f"=== Floor {floor_num + 1} ===\n"
                floor_specs += f"Rooms: {len(rooms)}\n"
                for room in rooms:
                    floor_specs += f"  Room: {room.w:.1f}×{room.d:.1f}×{room.h:.1f} - {room.color}\n"
                floor_specs += "\n"
            
            st.download_button(
                label="🏗️ Floor Specs",
                data=floor_specs,
                file_name="floor_specifications.txt",
                mime="text/plain",
                width='stretch'
            )
    
    with col_export4:
        if st.button("📸 View Guide", width='stretch'):
            st.info("Use browser screenshot or right-click on 3D view to save image")
    
    # ==================== FLOOR HIERARCHY VIEWER ====================
    with st.expander("🏢 View Floor Hierarchy"):
        st.markdown(f"**Floors:** {total_floors} | **Max Depth:** {st.session_state.max_depth}")
        st.code(print_floor_layout(st.session_state.root), language="text")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🏢 <strong>Vertical Floor Room Planner</strong> | Proper floor stacking with real wall gaps</p>
    <p><em>Exportable architectural models for multi-story building design</em></p>
</div>
""", unsafe_allow_html=True)
