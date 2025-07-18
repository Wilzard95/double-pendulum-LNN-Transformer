import numpy as np
import math
import cv2
import os
import time

# --- 1. Configuración Global y Parámetros Físicos (no cambian entre simulaciones) ---
L1 = 1.5; L2 = 1.0
m1 = 1.0; m2 = 5.0
g = 9.81

# Parámetros de la Simulación Numérica (aplican a cada simulación individual)
dt = 0.01
simulation_duration = 10.0 # Duración de cada simulación individual (ajustar según necesidad para el dataset)
                             # El artículo usa 10s para entrenamiento, 30s para test largo.

# Parámetros del Video y Datos de Salida (para UNA simulación con video, si se activa)
# La carpeta principal donde se guardará el dataset
main_output_folder = r"C:\Users\Nitro\Documents\Documentos William\Creando_mi_IA\Probando_modelos\Pendulos"
dataset_subfolder_name = "dataset_pendulo_doble_500" # Subcarpeta para los datos de las 500 simulaciones

# Parámetros de Visualización (solo si generamos video)
fps = 30 # Para video, si se genera
frame_width = 800
frame_height = 800
scale = 150
center_x, center_y = frame_width // 2, frame_height // 3
trace_length = 150
show_energy_in_video = True
show_velocity_vectors_in_video = True
vector_display_scale = 0.3
vector_thickness = 2
vector_tip_length = 0.2
color_m1 = (0, 255, 0); color_text_m1 = color_m1
color_m2 = (0, 0, 255); color_text_m2 = color_m2
color_lines = (255, 255, 255); color_text_total = color_lines
color_pivot = (128, 128, 128)
color_trace = (0, 100, 100)
color_vector = (0, 255, 255)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


# --- 2. Funciones Auxiliares (sin cambios) ---
def get_cartesian_coords(theta1, theta2, L1, L2):
    x1 = L1 * math.sin(theta1)
    y1 = L1 * math.cos(theta1)
    x2 = x1 + L2 * math.sin(theta2)
    y2 = y1 + L2 * math.cos(theta2)
    return x1, y1, x2, y2

def get_velocities(state, L1, L2):
    th1, w1, th2, w2 = state
    cos_th1 = math.cos(th1); sin_th1 = math.sin(th1)
    cos_th2 = math.cos(th2); sin_th2 = math.sin(th2)
    vx1 = L1 * cos_th1 * w1
    vy1 = -L1 * sin_th1 * w1
    vx2 = vx1 + L2 * cos_th2 * w2
    vy2 = vy1 - L2 * sin_th2 * w2
    return vx1, vy1, vx2, vy2

def derivatives(state, t, L1, L2, m1, m2, g):
    th1, w1, th2, w2 = state; dth1 = w1; dth2 = w2
    delta_th = th2 - th1; cos_delta = math.cos(delta_th); sin_delta = math.sin(delta_th)
    den1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2; den1 = max(abs(den1), 1e-8) * np.sign(den1) if den1 != 0 else 1e-8
    den2 = (L2 / L1) * den1; den2 = max(abs(den2), 1e-8) * np.sign(den2) if den2 != 0 else 1e-8
    dw1 = (m2*L1*w1**2*sin_delta*cos_delta + m2*g*math.sin(th2)*cos_delta + m2*L2*w2**2*sin_delta - (m1+m2)*g*math.sin(th1)) / den1
    dw2 = (-m2*L2*w2**2*sin_delta*cos_delta + (m1+m2)*g*math.sin(th1)*cos_delta - (m1+m2)*L1*w1**2*sin_delta - (m1+m2)*g*math.sin(th2)) / den2
    return np.array([dth1, dw1, dth2, dw2])

def rk4_step(state, t, dt, L1, L2, m1, m2, g):
    k1 = derivatives(state, t, L1, L2, m1, m2, g)
    k2 = derivatives(state + 0.5*dt*k1, t + 0.5*dt, L1, L2, m1, m2, g)
    k3 = derivatives(state + 0.5*dt*k2, t + 0.5*dt, L1, L2, m1, m2, g)
    k4 = derivatives(state + dt*k3, t + dt, L1, L2, m1, m2, g)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def calculate_energies(state, L1, L2, m1, m2, g, velocities=None):
    th1, w1, th2, w2 = state
    cos_th1 = math.cos(th1); cos_th2 = math.cos(th2)
    if velocities is None:
        vx1, vy1, vx2, vy2 = get_velocities(state, L1, L2)
    else:
        vx1, vy1, vx2, vy2 = velocities
    KE1 = 0.5 * m1 * (L1 * w1)**2
    v2_squared = vx2**2 + vy2**2
    KE2 = 0.5 * m2 * v2_squared
    T = KE1 + KE2
    PE1 = -m1 * g * L1 * cos_th1
    PE2 = -m2 * g * (L1 * cos_th1 + L2 * cos_th2)
    V = PE1 + PE2
    return KE1, KE2, T, V

# --- 3. Función para ejecutar UNA simulación y guardar datos/video ---
def run_single_simulation(sim_id, initial_state_radians, output_data_filepath,
                          L1, L2, m1, m2, g, dt, duration,
                          generate_video=False, video_fps=30, video_output_filepath=None):
    """
    Ejecuta una simulación del péndulo doble y guarda los datos.
    Opcionalmente, genera un video.
    """
    th1_rad_init, w1_rad_init, th2_rad_init, w2_rad_init = initial_state_radians
    state = np.array(initial_state_radians)

    # Convertir condiciones iniciales a grados para el encabezado del archivo
    th1_deg_init = math.degrees(th1_rad_init)
    th2_deg_init = math.degrees(th2_rad_init)
    w1_deg_s_init = math.degrees(w1_rad_init)
    w2_deg_s_init = math.degrees(w2_rad_init)

    video_writer = None
    if generate_video:
        if video_output_filepath is None:
            print("❌ ERROR: Se requiere video_output_filepath si generate_video es True.")
            return
        video_writer = cv2.VideoWriter(video_output_filepath, fourcc, video_fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"❌ ERROR: No se pudo abrir VideoWriter para {video_output_filepath}"); return
        # print(f"✅ VideoWriter iniciado para simulación {sim_id}. Guardando video en: {video_output_filepath}")

    try:
        with open(output_data_filepath, 'w') as data_file:
            # print(f"✅ Archivo de datos para simulación {sim_id} iniciado. Guardando datos en: {output_data_filepath}")
            data_file.write(f"# Simulation ID: {sim_id}\n")
            data_file.write("# --- Condiciones Iniciales y Parámetros Físicos ---\n")
            data_file.write(f"# L1={L1}, L2={L2}, m1={m1}, m2={m2}, g={g}\n")
            data_file.write(f"# th1_i={th1_deg_init:.4f}, th2_i={th2_deg_init:.4f} (deg)\n")
            data_file.write(f"# w1_i={w1_deg_s_init:.4f}, w2_i={w2_deg_s_init:.4f} (deg/s)\n")
            data_file.write(f"# dt={dt}, simulation_duration={duration}\n")
            data_file.write("# Coords: Origen pivote, x+ derecha, y+ ABAJO. Ángulos: 0=abajo, +antihorario (rad para estado, deg para log)\n")
            data_file.write("# Energías: V=0 en pivote (Joules).\n")
            data_file.write("# Velocidades cartesianas (vx, vy) en m/s.\n")
            data_file.write("# -----------------------------------------------\n\n")
            data_header = ("step\ttime_s\t"
                           "theta1_rad\tomega1_rad_s\ttheta2_rad\tomega2_rad_s\t" # Estado en radianes
                           "x1_m\ty1_m\tx2_m\ty2_m\t"
                           "vx1_mps\tvy1_mps\tvx2_mps\tvy2_mps\t"
                           "KE1_J\tKE2_J\tT_J\tV_J\tE_total_J\n") # Energía total
            data_file.write(data_header)

            total_steps = int(duration / dt)
            sim_time = 0.0
            trace_points = [] # Solo para video

            # El artículo guarda cada 'dt' (0.01s). Si queremos emular eso, no hay distinción entre sim_step y frame_step.
            # Si generamos video, podríamos querer submuestrear. Para datos, guardamos cada paso.
            
            steps_per_frame_for_video = int((1.0 / video_fps) / dt) if generate_video else 1
            data_step_counter = 0

            for step_idx in range(total_steps):
                state = rk4_step(state, sim_time, dt, L1, L2, m1, m2, g)
                sim_time += dt
                data_step_counter +=1

                th1_now, w1_now, th2_now, w2_now = state
                x1, y1, x2, y2 = get_cartesian_coords(th1_now, th2_now, L1, L2)
                vx1, vy1, vx2, vy2 = get_velocities(state, L1, L2)
                KE1, KE2, T, V = calculate_energies(state, L1, L2, m1, m2, g, velocities=(vx1, vy1, vx2, vy2))
                E_total = T + V

                data_line = (
                    f"{step_idx}\t{sim_time:.4f}\t"
                    f"{th1_now:.6f}\t{w1_now:.6f}\t{th2_now:.6f}\t{w2_now:.6f}\t"
                    f"{x1:.4f}\t{y1:.4f}\t{x2:.4f}\t{y2:.4f}\t"
                    f"{vx1:.4f}\t{vy1:.4f}\t{vx2:.4f}\t{vy2:.4f}\t"
                    f"{KE1:.4f}\t{KE2:.4f}\t{T:.4f}\t{V:.4f}\t{E_total:.4f}\n"
                )
                data_file.write(data_line)

                if generate_video and (step_idx % steps_per_frame_for_video == 0):
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    x1_px = int(center_x + x1 * scale); y1_px = int(center_y + y1 * scale)
                    x2_px = int(center_x + x2 * scale); y2_px = int(center_y + y2 * scale)

                    trace_points.append((x2_px, y2_px))
                    if len(trace_points) > trace_length: trace_points.pop(0)
                    if len(trace_points) > 1:
                        for i in range(len(trace_points) - 1):
                            cv2.line(frame, trace_points[i], trace_points[i+1], color_trace, 1)
                    cv2.line(frame, (center_x, center_y), (x1_px, y1_px), color_lines, 3)
                    cv2.line(frame, (x1_px, y1_px), (x2_px, y2_px), color_lines, 3)
                    cv2.circle(frame, (x1_px, y1_px), 10, color_m1, -1)
                    cv2.circle(frame, (x2_px, y2_px), 10, color_m2, -1)
                    cv2.circle(frame, (center_x, center_y), 5, color_pivot, -1)

                    if show_velocity_vectors_in_video:
                        end1_x_px = x1_px + int(vx1 * vector_display_scale * scale)
                        end1_y_px = y1_px + int(vy1 * vector_display_scale * scale)
                        end2_x_px = x2_px + int(vx2 * vector_display_scale * scale)
                        end2_y_px = y2_px + int(vy2 * vector_display_scale * scale)
                        cv2.arrowedLine(frame, (x1_px, y1_px), (end1_x_px, end1_y_px), color_vector, vector_thickness, tipLength=vector_tip_length)
                        cv2.arrowedLine(frame, (x2_px, y2_px), (end2_x_px, end2_y_px), color_vector, vector_thickness, tipLength=vector_tip_length)

                    time_text = f"Sim {sim_id} T: {sim_time:.2f}s E: {E_total:.1f}J"
                    cv2.putText(frame, time_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_lines, 1)
                    if show_energy_in_video:
                        y_text_start=30; line_height=25; font_scale=0.6; font_thickness=1
                        text_ke1=f"Tm_1: {KE1:.1f}J"; text_ke2=f"Tm_2: {KE2:.1f}J"; text_t=f"T: {T:.1f}J"; text_v=f"V: {V:.1f}J"
                        cv2.putText(frame, text_ke1, (10, y_text_start), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text_m1, font_thickness)
                        cv2.putText(frame, text_ke2, (10, y_text_start+line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text_m2, font_thickness)
                        cv2.putText(frame, text_t, (10, y_text_start+2*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text_total, font_thickness)
                        cv2.putText(frame, text_v, (10, y_text_start+3*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_text_total, font_thickness)
                    video_writer.write(frame)
            # print(f"   -> {total_steps} pasos de simulación procesados para sim {sim_id}.")

    except IOError as e: print(f"❌ ERROR escritura datos para sim {sim_id}: {output_data_filepath}\n   {e}")
    except Exception as e: print(f"❌ ERROR inesperado en sim {sim_id}: {e}")
    finally:
        if generate_video and video_writer is not None and video_writer.isOpened():
            video_writer.release()
            # print(f"✅ VideoWriter para simulación {sim_id} cerrado.")


# --- 4. Bucle Principal para Múltiples Simulaciones ---
if __name__ == '__main__':
    num_simulations_to_run = 500 # Objetivo: 500 simulaciones
    
    dataset_output_dir = os.path.join(main_output_folder, dataset_subfolder_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    print(f"--- Iniciando la generación de {num_simulations_to_run} simulaciones ---")
    print(f"Los datos se guardarán en: {dataset_output_dir}")

    overall_start_time = time.time()

    for i in range(num_simulations_to_run):
        sim_start_time = time.time()
        print(f"\n--- Empezando Simulación {i+1}/{num_simulations_to_run} ---")

        # Generar condiciones iniciales aleatorias (en RADIANES)
        # Según el artículo: q ~ U(-pi, pi), q_dot ~ U(-2, 2) rad/s
        theta1_initial_rad = np.random.uniform(-math.pi, math.pi)
        theta2_initial_rad = np.random.uniform(-math.pi, math.pi)
        omega1_initial_rad_s = np.random.uniform(-2.0, 2.0) # rad/s
        omega2_initial_rad_s = np.random.uniform(-2.0, 2.0) # rad/s
        
        current_initial_state_rad = np.array([
            theta1_initial_rad, omega1_initial_rad_s,
            theta2_initial_rad, omega2_initial_rad_s
        ])

        # Definir nombres de archivo para esta simulación específica
        # Datos siempre se guardan
        output_data_filename = f"sim_data_{i:03d}.txt" # e.g., sim_data_000.txt
        output_data_filepath = os.path.join(dataset_output_dir, output_data_filename)

        # Video es opcional (desactivado por defecto para el bucle masivo)
        # Si quisieras un video de la PRIMERA simulación, por ejemplo:
        # generate_this_video = (i == 0)
        generate_this_video = False # Desactivado para el bucle de 500
        video_output_filepath_this_sim = None
        if generate_this_video:
            video_filename = f"sim_video_{i:03d}.mp4"
            video_output_filepath_this_sim = os.path.join(dataset_output_dir, video_filename)
        
        run_single_simulation(
            sim_id=i,
            initial_state_radians=current_initial_state_rad,
            output_data_filepath=output_data_filepath,
            L1=L1, L2=L2, m1=m1, m2=m2, g=g,
            dt=dt, duration=simulation_duration,
            generate_video=generate_this_video, # Controla si se genera video
            video_fps=fps,
            video_output_filepath=video_output_filepath_this_sim
        )
        sim_end_time = time.time()
        print(f"Simulación {i+1} completada. Datos guardados en: {output_data_filepath}")
        if generate_this_video:
             print(f"Video para simulación {i+1} guardado en: {video_output_filepath_this_sim}")
        print(f"Tiempo para esta simulación: {sim_end_time - sim_start_time:.2f} segundos.")

    overall_end_time = time.time()
    print(f"\n--- Todas las {num_simulations_to_run} simulaciones completadas. ---")
    print(f"Tiempo total de ejecución: {overall_end_time - overall_start_time:.2f} segundos.")
    print(f"Los archivos de datos se encuentran en: {dataset_output_dir}")