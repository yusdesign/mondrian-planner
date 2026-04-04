import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
from scipy.spatial import cKDTree
import plotly.express as px

st.set_page_config(page_title="3D Mondrian kd-Tree Recursion", layout="wide")

st.title("🎨 3D Mondrian kd-Tree Recursion")
st.markdown("### Recursive axis-aligned splitting in 3D space")

# Color palettes for 3D volumes
COLOR_PALETTES = {
    'Mondrian': ['#FF0000', '#FFFF00', '#0000FF', '#FFFFFF', '#000000'],
    'Modern': ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2'],
    'Grayscale': ['#333333', '#666666', '#999999', '#CCCCCC'],
    'Vibrant': ['#E63946', '#F4A261', '#2A9D8F', '#E9C46A', '#264653']
}

class KDNode3D:
    """3D kd-Tree node for recursive space partitioning"""
    def __init__(self, x, y, z, w, d, h, depth, axis):
        self.x, self.y, self.z = x, y, z
        self.w, self.d, self.h = w, d, h  # width, depth, height
        self.depth = depth
        self.axis = axis  # 0:X, 1:Y, 2:Z
        self.children = []
        self.is_leaf = True
        self.color = None
        self.opacity = 0.8

def generate_3d_kdtree(x, y, z, w, d, h, depth, max_depth, axis, randomness, rng):
    """Recursively generate 3D kd-tree splits"""
    node = KDNode3D(x, y, z, w, d, h, depth, axis)
    
    if depth >= max_depth or min(w, d, h) < 15:
        return node
    
    # Split ratio between 0.25 and 0.75
    split_ratio = 0.3 + rng.random() * (0.7 - 0.3) * (1 - randomness)
    
    if axis == 0:  # Split along X
        split_x = x + w * split_ratio
        if split_x - x > 10 and (x + w) - split_x > 10:
            node.children.append(generate_3d_kdtree(x, y, z, split_x - x, d, h, 
                                                    depth + 1, max_depth, 1, randomness, rng))
            node.children.append(generate_3d_kdtree(split_x, y, z, x + w - split_x, d, h,
                                                    depth + 1, max_depth, 1, randomness, rng))
            node.is_leaf = False
    elif axis == 1:  # Split along Y
        split_y = y + d * split_ratio
        if split_y - y > 10 and (y + d) - split_y > 10:
            node.children.append(generate_3d_kdtree(x, y, z, w, split_y - y, h,
                                                    depth + 1, max_depth, 2, randomness, rng))
            node.children.append(generate_3d_kdtree(x, split_y, z, w, y + d - split_y, h,
                                                    depth + 1, max_depth, 2, randomness, rng))
            node.is_leaf = False
    else:  # Split along Z
        split_z = z + h * split_ratio
        if split_z - z > 10 and (z + h) - split_z > 10:
            node.children.append(generate_3d_kdtree(x, y, z, w, d, split_z - z,
                                                    depth + 1, max_depth, 0, randomness, rng))
            node.children.append(generate_3d_kdtree(x, y, split_z, w, d, z + h - split_z,
                                                    depth + 1, max_depth, 0, randomness, rng))
            node.is_leaf = False
    
    return node

def assign_colors_3d(node, density, balance, palette, rng):
    """Assign colors to 3D leaf nodes"""
    if node.is_leaf:
        volume = node.w * node.d * node.h
        color_prob = density * (1 + balance * (volume / (200**3)))
        if rng.random() < min(color_prob, 0.9):
            node.color = rng.choice(palette)
            node.opacity = 0.7 + rng.random() * 0.3
        else:
            node.color = '#FFFFFF' if '#FFFFFF' in palette else '#EEEEEE'
            node.opacity = 0.3
    else:
        for child in node.children:
            assign_colors_3d(child, density, balance, palette, rng)

def render_3d_scene(node, fig, show_lines=True):
    """Render 3D kd-tree as Plotly mesh3d"""
    if node.is_leaf:
        # Create vertices for cuboid
        x = [node.x, node.x + node.w, node.x + node.w, node.x, node.x, node.x + node.w, node.x + node.w, node.x]
        y = [node.y, node.y, node.y + node.d, node.y + node.d, node.y, node.y, node.y + node.d, node.y + node.d]
        z = [node.z, node.z, node.z, node.z, node.z + node.h, node.z + node.h, node.z + node.h, node.z + node.h]
        
        # Define faces (i,j,k indices for mesh)
        i = [0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3]
        j = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0]
        k = [2, 3, 0, 5, 6, 7, 4, 1, 5, 6, 7, 4]
        
        # Add mesh with opacity
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=node.color,
            opacity=node.opacity,
            name=f'Leaf d{node.depth}',
            showlegend=False,
            hoverinfo='text',
            text=f'Volume: {node.w}×{node.d}×{node.h}<br>Depth: {node.depth}'
        ))
        
        # Add wireframe edges
        if show_lines:
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            edge_x, edge_y, edge_z = [], [], []
            for start, end in edges:
                edge_x.extend([x[start], x[end], None])
                edge_y.extend([y[start], y[end], None])
                edge_z.extend([z[start], z[end], None])
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines', line=dict(color='black', width=2),
                showlegend=False, hoverinfo='none'
            ))
    else:
        # Draw split planes as semi-transparent surfaces
        if node.axis == 0 and show_lines:  # X split
            split_x = node.children[0].x + node.children[0].w
            y_range = np.linspace(node.y, node.y + node.d, 10)
            z_range = np.linspace(node.z, node.z + node.h, 10)
            Y, Z = np.meshgrid(y_range, z_range)
            X = np.full_like(Y, split_x)
            fig.add_surface(x=X, y=Y, z=Z, colorscale=[[0, 'rgba(100,100,100,0.2)'], [1, 'rgba(100,100,100,0.2)']], showscale=False)
        elif node.axis == 1 and show_lines:  # Y split
            split_y = node.children[0].y + node.children[0].d
            x_range = np.linspace(node.x, node.x + node.w, 10)
            z_range = np.linspace(node.z, node.z + node.h, 10)
            X, Z = np.meshgrid(x_range, z_range)
            Y = np.full_like(X, split_y)
            fig.add_surface(x=X, y=Y, z=Z, colorscale=[[0, 'rgba(100,100,100,0.2)'], [1, 'rgba(100,100,100,0.2)']], showscale=False)
        elif node.axis == 2 and show_lines:  # Z split
            split_z = node.children[0].z + node.children[0].h
            x_range = np.linspace(node.x, node.x + node.w, 10)
            y_range = np.linspace(node.y, node.y + node.d, 10)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.full_like(X, split_z)
            fig.add_surface(x=X, y=Y, z=Z, colorscale=[[0, 'rgba(100,100,100,0.2)'], [1, 'rgba(100,100,100,0.2)']], showscale=False)
        
        for child in node.children:
            render_3d_scene(child, fig, show_lines)

def count_leaves(node):
    """Count number of leaf nodes in kd-tree"""
    if not node: return 0
    if node.is_leaf: return 1
    return count_leaves(node.children[0]) + count_leaves(node.children[1])

def collect_colored(node):
    """Collect all colored leaf nodes"""
    if node.is_leaf and node.color not in ['#FFFFFF', '#EEEEEE']:
        yield node
    elif not node.is_leaf:
        for child in node.children:
            yield from collect_colored(child)

def print_tree_structure(node, level=0):
    """Return formatted string of kd-tree structure"""
    indent = "  " * level
    if node.is_leaf:
        color_info = node.color if node.color else "white"
        return f"{indent}📦 Leaf: {node.w}×{node.d}×{node.h} - {color_info}"
    axis_name = ["X", "Y", "Z"][node.axis]
    result = f"{indent}🔪 Split along {axis_name} (depth {node.depth})\n"
    result += print_tree_structure(node.children[0], level + 1) + "\n"
    result += print_tree_structure(node.children[1], level + 1)
    return result

# Sidebar controls
with st.sidebar:
    st.header("⚙️ 3D kd-Tree Parameters")
    
    canvas_size = st.slider("Canvas Size (px)", 100, 300, 200)
    max_depth = st.slider("Max Recursion Depth", 1, 5, 3)
    randomness = st.slider("Split Randomness", 0.0, 1.0, 0.3, 0.05)
    
    st.divider()
    
    color_density = st.slider("Color Density", 0.0, 1.0, 0.4, 0.05)
    color_balance = st.slider("Volume Balance (larger gets color)", 0.0, 1.0, 0.6)
    palette_name = st.selectbox("Color Palette", list(COLOR_PALETTES.keys()))
    
    st.divider()
    
    show_wireframe = st.checkbox("Show Wireframe Edges", True)
    show_split_planes = st.checkbox("Show Split Planes", True)
    
    seed = st.number_input("Random Seed", 0, 9999, 42)
    
    if st.button("🎲 Generate 3D kd-Tree", type="primary", width='stretch'):
        st.rerun()

# Generate 3D scene
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

# Generate root node (cube)
margin = 20
root = generate_3d_kdtree(margin, margin, margin, 
                         canvas_size - 2*margin, 
                         canvas_size - 2*margin,
                         canvas_size - 2*margin,
                         0, max_depth, 0, randomness, custom_rng)

assign_colors_3d(root, color_density, color_balance, palette, custom_rng)

# Create Plotly figure
fig = go.Figure()

render_3d_scene(root, fig, show_wireframe and show_split_planes)

# Update layout
fig.update_layout(
    title=f"3D Mondrian kd-Tree | Depth: {max_depth} | Leaves: {count_leaves(root)}",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis", 
        zaxis_title="Z Axis",
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        bgcolor='#f3f2ee'
    ),
    paper_bgcolor='#f3f2ee',
    margin=dict(l=0, r=0, t=50, b=0),
    showlegend=False
)

# Display
st.plotly_chart(fig, use_container_width=True)

# Stats columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌲 Leaf Nodes", count_leaves(root))
with col2:
    st.metric("🔪 Max Depth", max_depth)
with col3:
    st.metric("📦 Total Volume", f"{canvas_size}³")
with col4:
    colored = sum(1 for _ in collect_colored(root))
    st.metric("🎨 Colored Volumes", colored)

# kd-Tree structure display
with st.expander("📊 View kd-Tree Structure"):
    st.code(print_tree_structure(root), language="text")

st.markdown("""
### 🌟 3D Mondrian kd-Tree Properties

- **Recursive 3D splitting** alternates between X, Y, and Z axes
- Each **leaf node** is a rectangular cuboid (3D rectangle)
- **Split planes** create hierarchical space partitioning
- **Color intensity** based on volume (larger volumes more likely colored)
- Perfect for **3D visualization**, **voxel art**, and **spatial data structures**

*Inspired by Piet Mondrian's geometric abstraction and kd-tree algorithms*
""")
