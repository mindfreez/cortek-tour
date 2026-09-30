# Gate-scene car: a Model Y-inspired crossover in pearl white on 20" rims, built procedurally in Blender and exported to car-modely.glb.
# Run inside Blender (Text editor > Run Script, or `blender --background --python car_modely.py`).
# Blender axes: X = width (driver side is -X), Y = forward (nose toward +Y), Z = up. The glTF export turns
# +Y into three.js -Z, so the nose points toward -z in the page, like the procedural car it replaces.
# Material names (Paint, Glass, Trim, Tire, Rim, LampFront, LampRear, Plate, Interior, RoofGlass) are swapped for the page's own
# materials at load time, so colours here only matter for previews.
import bpy, bmesh, math, os
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else r"C:\AI Workspace\cortek-tour\tools\blender"
OUT_GLB = os.path.normpath(os.path.join(HERE, '..', '..', 'car-modely.glb'))
COLL = 'CarModelY'

def pchip(pts):
    """Monotone cubic through (x, y) control points: smooth, never overshoots."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]; n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]; d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [d[0]] + [0.0] * (n - 2) + [d[-1]]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    def f(x):
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        i = max(j for j in range(n - 1) if xs[j] <= x); t = (x - xs[i]) / h[i]
        return ((2*t**3 - 3*t**2 + 1) * ys[i] + (t**3 - 2*t**2 + t) * h[i] * m[i]
                + (-2*t**3 + 3*t**2) * ys[i + 1] + (t**3 - t**2) * h[i] * m[i + 1])
    return f

# ---- side profile and plan (metres); car is 4.5 m long, 1.87 m wide, 1.52 m tall ----
ZT = pchip([(-2.25, .8), (-2.23, .94), (-2.18, 1.02), (-2.08, 1.06), (-1.85, 1.15), (-1.35, 1.33), (-.8, 1.46), (-.3, 1.52),
            (.15, 1.51), (.5, 1.44), (.9, 1.24), (1.3, 1.0), (1.65, .92), (1.95, .85), (2.12, .77), (2.2, .68), (2.25, .56)])  # roofline
ZB = pchip([(-2.25, .52), (-2.18, .37), (-1.95, .21), (1.9, .21), (2.15, .3), (2.25, .44)])                                        # underside
ZBELT = pchip([(-2.25, .98), (-2.0, 1.0), (-1.0, 1.01), (0, .99), (1.0, .96), (1.3, .95), (2.25, .88)])                            # window line
HW = pchip([(-2.25, .46), (-2.23, .66), (-2.17, .79), (-2.0, .88), (-1.6, .925), (-.5, .935), (1.2, .93), (1.8, .9), (2.05, .85),
            (2.17, .76), (2.23, .63), (2.25, .46)])                                                                                 # half width
WHEELBASE, WHEEL_R, TRACK = 2.76, .36, .81            # 255/40 R20: 0.51 m rim inside a 0.72 m tyre
RIM_R = .262

def clamp01(v): return max(0.0, min(1.0, v))

def half_ring(y):
    """Cross-section at station y, right side only: bottom centre -> side -> top centre, as (x, z)."""
    zb, zt, hw = ZB(y), ZT(y), HW(y); zs = min(ZBELT(y), zt - .08)
    pts = [(0, zb), (.55 * hw, zb), (.78 * hw, zb + .005), (.88 * hw, zb + .025), (.94 * hw, zb + .06), (.975 * hw, zb + .11)]
    for f, w in ((.2, .99), (.45, 1.0), (.7, .995), (.88, .98)):   # softly curved flanks, widest just below the middle
        pts.append((w * hw, zb + .11 + f * (zs - zb - .11)))
    b = zt - zs; g = clamp01((b - .12) / .3)                  # g: 0 over the hood/deck, 1 over the greenhouse
    e, T, a, M = .7 - .25 * g, .34 * g, .965 * hw, 16          # superellipse exponent (higher = rounder), tumblehome, width, steps
    for j in range(M + 1):
        th = j / M * math.pi / 2
        x = a * max(math.cos(th), 0) ** e; z = zs + b * math.sin(th) ** e
        pts.append((0.0 if j == M else x * (1 - T * (z - zs) / max(b, 1e-4)), z))
    return pts

def material(name, rgb, metal=0.0, rough=.5, alpha=1.0, emit=None):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    p = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if p:
        p.inputs['Base Color'].default_value = (*rgb, 1); p.inputs['Metallic'].default_value = metal
        p.inputs['Roughness'].default_value = rough; p.inputs['Alpha'].default_value = alpha
        if emit:
            p.inputs['Emission Color'].default_value = (*emit, 1); p.inputs['Emission Strength'].default_value = 2.0
    m.diffuse_color = (*(emit or rgb), alpha); m.metallic = metal; m.roughness = rough
    return m

def link(obj, parent):
    bpy.data.collections[COLL].objects.link(obj); obj.parent = parent; return obj

def mesh_obj(name, verts, faces, mats, parent):
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    ob = bpy.data.objects.new(name, me)
    for m in mats: me.materials.append(m)
    return link(ob, parent)

def smooth(ob, angle=35):
    for p in ob.data.polygons: p.use_smooth = True
    try: ob.data.set_sharp_from_angle(angle=math.radians(angle))
    except Exception: pass

def prim(op, name, mat, parent, **kw):
    getattr(bpy.ops.mesh, op)(**kw); ob = bpy.context.active_object; ob.name = name
    for c in ob.users_collection: c.objects.unlink(ob)
    ob.data.materials.append(mat); return link(ob, parent)

def apply_mods(ob):
    dg = bpy.context.evaluated_depsgraph_get(); me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    ob.modifiers.clear(); old = ob.data; ob.data = me; bpy.data.meshes.remove(old)

def build():
    # fresh collection each run
    if COLL in bpy.data.collections:
        for o in list(bpy.data.collections[COLL].objects): bpy.data.objects.remove(o, do_unlink=True)
    else:
        bpy.context.scene.collection.children.link(bpy.data.collections.new(COLL))
    cube = bpy.data.objects.get('Cube')
    if cube: bpy.data.objects.remove(cube, do_unlink=True)

    M = dict(Paint=material('Paint', (.86, .85, .8), .1, .25), Glass=material('Glass', (.006, .01, .014), .7, .05, .62),
             Trim=material('Trim', (.008, .009, .011), 0, .6), Tire=material('Tire', (.012, .012, .013), 0, .85),
             Rim=material('Rim', (.045, .048, .055), .7, .3), LampFront=material('LampFront', (1, 1, 1), emit=(.9, .93, 1)),
             LampRear=material('LampRear', (.8, .02, .03), emit=(.9, .03, .05)), Plate=material('Plate', (.9, .9, .88), 0, .35),
             Interior=material('Interior', (.012, .012, .015), 0, .9), RoofGlass=material('RoofGlass', (.006, .01, .014), .7, .08))

    root = bpy.data.objects.new('CarModelY', None); bpy.data.collections[COLL].objects.link(root)

    # ---- body: loft of cross-sections along Y, capped at both ends ----
    ys = sorted({round(-2.25 + i * 4.5 / 64, 4) for i in range(65)} | {round(s * (2.05 + i * .02), 4) for i in range(11) for s in (1, -1)})
    rings = []
    for y in ys:
        r = half_ring(y); rings.append([(x, y, z) for x, z in r] + [(-x, y, z) for x, z in reversed(r[1:-1])])
    R = len(rings[0]); verts = [v for ring in rings for v in ring]
    faces = [(s * R + k, s * R + (k + 1) % R, (s + 1) * R + (k + 1) % R, (s + 1) * R + k) for s in range(len(ys) - 1) for k in range(R)]
    faces += [tuple(range(R))[::-1], tuple((len(ys) - 1) * R + k for k in range(R))]
    body = mesh_obj('Body', verts, faces, [M['Paint'], M['Glass'], M['Trim'], M['RoofGlass']], root)
    bm = bmesh.new(); bm.from_mesh(body.data); bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:                                          # glass above the window line (opaque on top), black underside
        c = f.calc_center_median(); zbl = ZBELT(c.y)
        if c.z > zbl + .012 and ZT(c.y) - zbl > .07 and -2.12 < c.y < 1.42 and f.normal.z > -.2: f.material_index = 3 if f.normal.z > .8 else 1
        elif f.normal.z < -.6 or c.z < ZB(c.y) + .035: f.material_index = 2
    bm.to_mesh(body.data); bm.free()

    # wheel arches: cut short cylinders out of each side (their walls become black arch liners)
    cut = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(prim('primitive_cylinder_add', 'ArchCut', M['Trim'], root, vertices=48, radius=WHEEL_R + .04, depth=.5,
                            location=(sx * .85, sy * WHEELBASE / 2, WHEEL_R), rotation=(0, math.pi / 2, 0)))
    for c in cut:
        mod = body.modifiers.new('arch', 'BOOLEAN'); mod.operation = 'DIFFERENCE'; mod.object = c; mod.solver = 'EXACT'
        try: mod.material_mode = 'TRANSFER'
        except Exception: pass
        apply_mods(body); bpy.data.objects.remove(c, do_unlink=True)
    smooth(body, 50)

    # ---- light bars, intake, plate: strips that hug the body, placed by ray casts ----
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    def strip(name, mat, centre, radius, phis, z, half_h, off=.005):
        vs, fs, ns = [], [], []
        for phi in phis:
            o = Vector((centre[0] + math.sin(phi) * radius, centre[1] + math.cos(phi) * radius, z))
            hit, nrm, _, _ = bvh.ray_cast(o, (Vector((centre[0], centre[1], z)) - o).normalized())
            if hit is None: continue
            t = (Vector((0, 0, 1)) - nrm * nrm.z).normalized(); p = hit + nrm * off
            vs += [p - t * half_h, p + t * half_h]; ns.append(nrm)
        for i in range(len(vs) // 2 - 1):
            a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
            n = (Vector(vs[b]) - Vector(vs[a])).cross(Vector(vs[d]) - Vector(vs[a]))
            fs.append((a, b, c, d) if n.dot(ns[i]) > 0 else (a, d, c, b))
        ob = mesh_obj(name, [tuple(v) for v in vs], fs, [mat], root); smooth(ob, 60); return ob
    rng = lambda a, b, n: [math.radians(a + (b - a) * i / (n - 1)) for i in range(n)]
    strip('LightBarFront', M['LampFront'], (0, 1.55), 3, rng(-54, 54, 65), .72, .01)
    for s in (-1, 1): strip('Headlight', M['LampFront'], (0, 1.55), 3, rng(s * 22, s * 40, 14), .65, .018)
    strip('Intake', M['Trim'], (0, 1.55), 3, rng(-20, 20, 21), .42, .045, .003)
    strip('LightBarRear', M['LampRear'], (0, -1.55), 3, [math.pi + p for p in rng(-60, 60, 75)], .88, .015)
    strip('Diffuser', M['Trim'], (0, -1.55), 3, [math.pi + p for p in rng(-30, 30, 25)], .56, .06, .003)
    strip('Plate', M['Plate'], (0, -1.55), 3, [math.pi + p for p in rng(-5.5, 5.5, 5)], .68, .075, .008)

    # ---- cabin: a dark tub up to the window line plus seat backs, so the tinted glass doesn't show an empty shell.
    #      The page's driver sphere sits at (-.4, -.25, 1.12) here, above the tub and in front of the seat back. ----
    for name, loc, scale in (('Tub', (0, -.35, .635), (1.6, 3.1, .67)),
                             ('SeatFront', (-.4, -.55, 1.15), (.46, .1, .36)), ('SeatFront', (.4, -.55, 1.15), (.46, .1, .36)),
                             ('SeatRear', (0, -1.4, 1.08), (1.3, .1, .26))):
        s = prim('primitive_cube_add', name, M['Interior'], root, size=1, location=loc); s.scale = scale

    # ---- mirrors on the doors, just behind the A-pillars: small rounded pods ----
    for sx in (-1, 1):
        m = prim('primitive_uv_sphere_add', 'Mirror', M['Paint'], root, segments=20, ring_count=12, radius=1, location=(sx * .99, .98, 1.02))
        m.scale = (.1, .055, .05); smooth(m, 80)

    # ---- wheels: low-profile 255/40 R20 tyre, 20" ten-spoke rim (twisted spokes over a dark dish), hub ----
    for sx in (-1, 1):
        for sy in (-1, 1):
            loc = (sx * TRACK, sy * WHEELBASE / 2, WHEEL_R); face = sx * (TRACK + .1)
            # tyre: a torus squashed along the axle, so it's a ring (the rim shows through) with rounded shoulders
            t = prim('primitive_torus_add', 'Tire', M['Tire'], root, major_radius=(WHEEL_R + RIM_R) / 2, minor_radius=(WHEEL_R - RIM_R) / 2,
                     major_segments=48, minor_segments=12, location=loc, rotation=(0, math.pi / 2, 0))
            t.scale = (1, 1, .255 / (WHEEL_R - RIM_R)); smooth(t, 80)
            prim('primitive_cylinder_add', 'Dish', M['Trim'], root, vertices=48, radius=RIM_R, depth=.02, location=(face - sx * .07, loc[1], WHEEL_R), rotation=(0, math.pi / 2, 0))
            lip = prim('primitive_torus_add', 'RimLip', M['Rim'], root, major_radius=RIM_R - .006, minor_radius=.01, major_segments=48, minor_segments=8,
                       location=(face - sx * .012, loc[1], WHEEL_R), rotation=(0, math.pi / 2, 0)); smooth(lip, 80)
            for k in range(10):                                 # spokes run from the hub out to the lip, dished inward
                a = k * 2 * math.pi / 10
                s = prim('primitive_cube_add', 'Spoke', M['Rim'], root, size=1, location=(face - sx * .03, loc[1] + math.sin(a) * .155, WHEEL_R + math.cos(a) * .155),
                         rotation=(-a, 0, 0))
                s.scale = (.014, .028, .2)
            prim('primitive_cylinder_add', 'Hub', M['Rim'], root, vertices=24, radius=.065, depth=.04, location=(face - sx * .03, loc[1], WHEEL_R), rotation=(0, math.pi / 2, 0))
            prim('primitive_cylinder_add', 'Cap', M['Trim'], root, vertices=20, radius=.035, depth=.01, location=(face - sx * .008, loc[1], WHEEL_R), rotation=(0, math.pi / 2, 0))

def export():
    bpy.ops.object.select_all(action='DESELECT')
    for o in bpy.data.collections[COLL].objects: o.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['CarModelY']
    bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format='GLB', use_selection=True, export_apply=True, export_yup=True)

def preview(out_dir):
    """Quick Workbench renders from three angles, for review."""
    sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'MATERIAL'; sc.display.shading.show_shadows = True; sc.display.shading.show_cavity = True
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720; sc.render.film_transparent = False
    cam = bpy.data.objects.get('PreviewCam')
    if not cam:
        cam = bpy.data.objects.new('PreviewCam', bpy.data.cameras.new('PreviewCam')); sc.collection.objects.link(cam)
    cam.data.lens = 50; sc.camera = cam
    out = []
    for name, pos in (('front34', (4.6, 5.6, 1.9)), ('rear34', (-4.4, -5.8, 2.1)), ('side', (-8.5, 0, 1.0))):
        cam.location = pos; d = Vector((0, 0, .75)) - Vector(pos); cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(out_dir, f'car-modely-{name}.png'); bpy.ops.render.render(write_still=True); out.append(sc.render.filepath)
    return out

if __name__ == '__main__':
    build(); export()
