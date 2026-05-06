import streamlit as st
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
import io
import base64
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="3D Mondrian kd-Tree Recursion",
    page_icon="🎨",
    layout="wide"
)

# ==================== SOCIAL SHARING ====================
st.markdown("""
    <meta property="og:title" content="3D Mondrian kd-Tree Recursion">
    <meta property="og:description" content="Interactive 3D visualization of recursive axis-aligned splitting">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://mondrian-planner.streamlit.app">
    <meta property="og:image" content="https://raw.githubusercontent.com/yusdesign/mondrian-planner/main/og_image.png">
    <meta name="twitter:card" content="summary_large_image">
""", unsafe_allow_html=True)

st.title("🎨 3D Mondrian kd-Tree Recursion")
st.markdown("### *Recursive axis-aligned splitting in 3D space | No triangulation artifacts*")

# ==================== COLOR PALETTES ====================
COLOR_PALETTES = {
    'Mondrian': ['#FF0000', '#FFFF00', '#0000FF', '#FFFFFF', '#000000'],
    'Modern': ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2'],
    'Grayscale': ['#333333', '#666666', '#999999', '#CCCCCC'],
    'Vibrant': ['#E63946', '#F4A261', '#2A9D8F', '#E9C46A', '#264653'],
    'Pastel': ['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA', '#FFB3BA']
}

# ==================== KD-TREE CLASS ====================
class KDNode3D:
    def __init__(self, x, y, z, w, d, h, depth, axis):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h
        self.depth = depth
        self.axis = axis
        self.children = []
        self.is_leaf = True
        self.color = None
        self.opacity = 0.8

def generate_3d_kdtree(x, y, z, w, d, h, depth, max_depth, axis, randomness, rng):
    node = KDNode3D(x, y, z, w, d, h, depth, axis)
    
    if depth >= max_depth or min(w, d, h) < 15:
        return node
    
    split_ratio = 0.3 + rng.random() * (0.7 - 0.3) * (1 - randomness)
    
    if axis == 0:
        split_x = x + w * split_ratio
        if split_x - x > 10 and (x + w) - split_x > 10:
            node.children.append(generate_3d_kdtree(x, y, z, split_x - x, d, h, 
                                                    depth + 1, max_depth, 1, randomness, rng))
            node.children.append(generate_3d_kdtree(split_x, y, z, x + w - split_x, d, h,
                                                    depth + 1, max_depth, 1, randomness, rng))
            node.is_leaf = False
    elif axis == 1:
        split_y = y + d * split_ratio
        if split_y - y > 10 and (y + d) - split_y > 10:
            node.children.append(generate_3d_kdtree(x, y, z, w, split_y - y, h,
                                                    depth + 1, max_depth, 2, randomness, rng))
            node.children.append(generate_3d_kdtree(x, split_y, z, w, y + d - split_y, h,
                                                    depth + 1, max_depth, 2, randomness, rng))
            node.is_leaf = False
    else:
        split_z = z + h * split_ratio
        if split_z - z > 10 and (z + h) - split_z > 10:
            node.children.append(generate_3d_kdtree(x, y, z, w, d, split_z - z,
                                                    depth + 1, max_depth, 0, randomness, rng))
            node.children.append(generate_3d_kdtree(x, y, split_z, w, d, z + h - split_z,
                                                    depth + 1, max_depth, 0, randomness, rng))
            node.is_leaf = False
    
    return node

def assign_colors_3d(node, density, balance, palette, rng):
    if node.is_leaf:
        volume = node.w * node.d * node.h
        color_prob = density * (1 + balance * (volume / (200**3)))
        if rng.random() < min(color_prob, 0.9):
            node.color = rng.choice(palette)
            node.opacity = 0.8
        else:
            node.color = '#FFFFFF' if '#FFFFFF' in palette else '#EEEEEE'
            node.opacity = 0.3
    else:
        for child in node.children:
            assign_colors_3d(child, density, balance, palette, rng)

def add_cuboid(fig, x, y, z, w, d, h, color, opacity, line_width=1):
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
            line=dict(color='black', width=line_width),
            showlegend=False, hoverinfo='none'
        ))

def render_3d_scene(node, fig, show_lines=True):
    if node.is_leaf:
        add_cuboid(fig, node.x, node.y, node.z, node.w, node.d, node.h, 
                  node.color, node.opacity, 2 if show_lines else 0)
    else:
        for child in node.children:
            render_3d_scene(child, fig, show_lines)

def count_leaves(node):
    if not node: return 0
    if node.is_leaf: return 1
    return count_leaves(node.children[0]) + count_leaves(node.children[1])

def collect_colored(node):
    if node.is_leaf and node.color not in ['#FFFFFF', '#EEEEEE']:
        yield node
    elif not node.is_leaf:
        for child in node.children:
            yield from collect_colored(child)

def print_tree_structure(node, level=0):
    indent = "  " * level
    if node.is_leaf:
        color_info = node.color if node.color else "white"
        return f"{indent}📦 Leaf: {node.w}×{node.d}×{node.h} - {color_info}"
    axis_name = ["X", "Y", "Z"][node.axis]
    result = f"{indent}🔪 Split along {axis_name} (depth {node.depth})\n"
    result += print_tree_structure(node.children[0], level + 1) + "\n"
    result += print_tree_structure(node.children[1], level + 1)
    return result

def export_to_obj(node):
    """Export kd-tree to OBJ file format"""
    vertices = []
    faces = []
    
    def collect_mesh(node):
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
            
            # Each face as two triangles (for better compatibility)
            face_indices = [
                [base_idx, base_idx+1, base_idx+2], [base_idx, base_idx+2, base_idx+3],  # bottom
                [base_idx+4, base_idx+5, base_idx+6], [base_idx+4, base_idx+6, base_idx+7],  # top
                [base_idx, base_idx+1, base_idx+5], [base_idx, base_idx+5, base_idx+4],  # front
                [base_idx+2, base_idx+3, base_idx+7], [base_idx+2, base_idx+7, base_idx+6],  # back
                [base_idx, base_idx+3, base_idx+7], [base_idx, base_idx+7, base_idx+4],  # left
                [base_idx+1, base_idx+2, base_idx+6], [base_idx+1, base_idx+6, base_idx+5]   # right
            ]
            faces.extend(face_indices)
        else:
            for child in node.children:
                collect_mesh(child)
    
    collect_mesh(node)
    
    obj_content = "# 3D Mondrian kd-Tree\n"
    obj_content += f"o Mondrian_kdTree\n\n"
    
    for v in vertices:
        obj_content += f"v {v[0]} {v[1]} {v[2]}\n"
    
    obj_content += "\n"
    for face in faces:
        obj_content += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    
    return obj_content

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ 3D kd-Tree Parameters")
    
    canvas_size = st.slider("Canvas Size (px)", 100, 300, 200)
    max_depth = st.slider("Max Recursion Depth", 1, 5, 3)
    randomness = st.slider("Split Randomness", 0.0, 1.0, 0.3, 0.05)
    
    st.divider()
    
    color_density = st.slider("Color Density", 0.0, 1.0, 0.4, 0.05)
    color_balance = st.slider("Volume Balance", 0.0, 1.0, 0.6)
    palette_name = st.selectbox("Color Palette", list(COLOR_PALETTES.keys()))
    
    st.divider()
    
    show_wireframe = st.checkbox("Show Wireframe Edges", True)
    seed = st.number_input("Random Seed", 0, 9999, 42)
    
    generate_clicked = st.button("🎲 Generate 3D kd-Tree", type="primary", width='stretch')

# ==================== INITIALIZE SESSION STATE ====================
if 'root' not in st.session_state:
    st.session_state.root = None
if 'generated' not in st.session_state:
    st.session_state.generated = False

# ==================== GENERATE TREE ====================
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
    root = generate_3d_kdtree(margin, margin, margin, 
                             canvas_size - 2*margin, 
                             canvas_size - 2*margin,
                             canvas_size - 2*margin,
                             0, max_depth, 0, randomness, custom_rng)
    
    assign_colors_3d(root, color_density, color_balance, palette, custom_rng)
    
    st.session_state.root = root
    st.session_state.generated = True
    st.session_state.canvas_size = canvas_size
    st.session_state.max_depth = max_depth
    st.session_state.show_wireframe = show_wireframe

# ==================== RENDER SCENE ====================
if st.session_state.root:
    fig = go.Figure()
    render_3d_scene(st.session_state.root, fig, st.session_state.show_wireframe)
    
    fig.update_layout(
        title=f"3D Mondrian kd-Tree | Depth: {st.session_state.max_depth} | Leaves: {count_leaves(st.session_state.root)}",
        scene=dict(
            xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis",
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
    
    # ==================== STATS ====================
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🌲 Leaf Nodes", count_leaves(st.session_state.root))
    with col2:
        st.metric("🔪 Max Depth", st.session_state.max_depth)
    with col3:
        st.metric("📦 Total Volume", f"{st.session_state.canvas_size}³")
    with col4:
        colored = sum(1 for _ in collect_colored(st.session_state.root))
        st.metric("🎨 Colored", colored)
    with col5:
        ratio = colored / count_leaves(st.session_state.root) if count_leaves(st.session_state.root) > 0 else 0
        st.metric("🎯 Density", f"{ratio:.0%}")
    
    # ==================== EXPORT SECTION ====================
    st.divider()
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📸 Screenshot Guide", width='stretch'):
            st.info("Use your browser's screenshot tool or right-click on the 3D view above and select 'Save as image'")
    
    with col_export2:
        # OBJ Export button with proper data
        obj_data = export_to_obj(st.session_state.root)
        st.download_button(
            label="📦 Export as OBJ",
            data=obj_data,
            file_name="mondrian_3d.obj",
            mime="text/plain",
            width='stretch'
        )
    
    with col_export3:
        # Also export as JSON for the tree structure
        st.download_button(
            label="📊 Export Tree Structure",
            data=print_tree_structure(st.session_state.root),
            file_name="mondrian_tree.txt",
            mime="text/plain",
            width='stretch'
        )
    
    # ==================== TREE STRUCTURE ====================
    with st.expander("📊 View kd-Tree Structure"):
        st.code(print_tree_structure(st.session_state.root), language="text")

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌟 <strong>3D Mondrian kd-Tree</strong> | Recursive axis-aligned splitting | No triangulation artifacts</p>
    <p><em>Inspired by Piet Mondrian's geometric abstraction and kd-tree algorithms</em></p>
</div>
""", unsafe_allow_html=True)
