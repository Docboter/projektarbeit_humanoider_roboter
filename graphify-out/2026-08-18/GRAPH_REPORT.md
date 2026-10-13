# Graph Report - projektarbeit_humanoider_roboter  (2026-08-18)

## Corpus Check
- 317 files · ~951,306 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 5098 nodes · 8367 edges · 342 communities (312 shown, 30 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 360 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `29d03e49`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- handle
- cast.h
- cpp_function
- class.h
- init.h
- pybind11.h
- server_rl_run.sh
- Sim-Eval Implementation Notes — Lessons Learned & aktueller Stand
- Anleitung — GR00T N1.6 Fine-tuning
- GR00T N1.6 — Unitree G1 + DEX3 Setup-Dokumentation
- object
- render_cotrain_dataset.py
- class_
- test_eigen.py
- Closed-Loop-Sim auf vast.ai — Schritt-für-Schritt-Anleitung
- G1Dex3BlockstackEnv
- array
- operator<
- sequence
- doc
- convert.h
- numpy.h
- test_tagbased_polymorphic.cpp
- GR00T N1.6 Fine-tuning — Unitree G1 + DEX3 Hand
- UDP
- constructor_stats.h
- msg
- test_class.py
- LiveView
- UnitreeJointController
- T
- vector
- multiThread
- type_info
- rl_finetune.py
- type_caster_base
- test_builtin_casters.py
- PolicyClient
- load_impl
- test_copy_move.cpp
- function_record
- test_stl.py
- return_value_policy
- detail/common.h
- OstreamRedirect
- dtype
- string
- PyArrayDescr_Proxy
- test_methods_and_attributes.py
- test_numpy_dtypes.py
- lib_resume_guard.sh
- attr.h
- buffer_info
- G1Dex3BlockstackEnvCfg
- iostream
- UnitreeDrawForcePlugin
- TEST_SUBMODULE
- Auswertung — erster vollständiger Trainingsdurchlauf (`g1_dex3_blockstacking_v1`)
- UnitreeFootContactPlugin
- test_custom_type_casters.cpp
- Custom
- str
- print_destroyed
- convert_v3_to_v2_standalone.py
- unitree_legged_sdk.h
- type_record
- unpacking_collector
- conftest.py
- Implementierungsplan — Live-Ansicht der Isaac-Lab-Sim
- pybind11_tests.h
- HasOpNewDel
- pybind11_tests
- ExampleMandA
- test_multiple_inheritance.cpp
- test_numpy_array.py
- 3. Ablauf
- type_caster_generic
- env.py
- Pet
- test_class.cpp
- test_factory_constructors.cpp
- test_gil_scoped.py
- Lokomotion des Unitree G1 freischalten — Recherche
- server_robocasa_ref_run.sh
- Reinforcement-Learning-Training — Recherche & Umsetzungsplan
- stl
- complex.h
- teleForceCmd
- internals
- EigenProps
- test_local_bindings.py
- test_multiple_inheritance.py
- test_virtual_functions.cpp
- Auswertung — dritter Trainingsdurchlauf, Vision-Encoder **mit** Split & Augmentierung (`g1_dex3_blockstacking_vision_v2`)
- G1GripperBlockstackEnv
- Loop
- chrono.h
- test_stl_binders.cpp
- npy_api
- UserType
- test_eigen.cpp
- Journey Into projektarbeit_humanoider_roboter
- dump_camera_poses.py
- test_numpy_array.cpp
- Capture
- test_numpy_dtypes.cpp
- Domain-Gap-Analyse: Real → Isaac Sim
- Umgebungsanalyse: Training & Simulation
- PYBIND11_OVERRIDE
- simulation/README.md
- test_constants_and_functions.cpp
- Custom
- Custom
- Custom
- index_sequence
- pytypes.h
- pybind11_tests.cpp
- stl.h
- Pet
- test_interpreter.cpp
- test_methods_and_attributes.cpp
- test_operator_overloading.cpp
- RoboCasa GR-1 Referenz-Eval — Bedienung
- Basismodell-Fähigkeiten & Referenzaufgabe zur Sim-Validierung
- Live-Ansicht — Isaac Sim auf dem eigenen Rechner öffnen
- Schritt 3 — BC-Erfolgsrate in der Sim (`server_rl_run.sh eval`)
- Alloc
- different_resolutions
- CLAUDE.md
- multi_array_iterator
- type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
- Custom
- TEST_SUBMODULE
- function_call
- type_info
- iterator
- test_iostream.py
- test_sequences_and_iterators.cpp
- test_sequences_and_iterators.py
- test_smart_ptr.py
- test_stl.cpp
- test_virtual_functions.py
- Sim-Container auf KISSKI starten — Schritt-für-Schritt-Anleitung
- HPC-Training auf KISSKI
- Object
- joint_controller.cpp
- Custom
- Time
- type_caster<
    Eigen::Ref<PlainObjectType, 0, StrideType>,
    enable_if_t<is_eigen_dense_map<Eigen::Ref<PlainObjectType, 0, StrideType>>::value>
>
- test_chrono.py
- test_copy_move.py
- type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>>
- Development of pybind11
- process_attribute_default
- PYBIND11_OVERRIDE_PURE
- Auswertung — zweiter Trainingsdurchlauf mit Vision-Encoder (`g1_dex3_blockstacking_vision_v1`)
- Training
- Pfad A — vast.ai (Cloud-Miete)
- run_robocasa_ref_eval.sh
- entrypoint_baseline.sh
- entrypoint_sim.sh
- measure_domain_gap.py
- is_copy_assignable<std::pair<T1, T2>>
- TEST_SUBMODULE
- Detail of Packages
- Safety
- T
- any_container
- instance
- unchecked_reference
- test_buffers.py
- Closed-Loop-Simulation für G1 + Dex3 in Isaac Lab — Implementierungsplan
- Train-Test-Split (80/20)
- Was für ein Training wird hier durchgeführt?
- Troubleshooting
- ._force_camera_prim_orientations
- run_g1_gripper_sim_eval.py
- PolicyClient
- optimize_groot_inference.py
- eigen.h
- setup_helpers.py
- test_callbacks.py
- test_exceptions.py
- pybind11_fail
- Zweiten Docker-Container für die Sim bauen — Isaac Lab + GR00T-Client
- Multi-GPU-Training (bis zu 4× A100)
- 3. Priorisierte Schritte
- entrypoint_replay.sh
- main
- GR00T N1.6 — Unitree G1 + DEX3 Dexterous Hand
- Type
- UnitreeJointController::update
- unitree_ros_to_real/README.md
- test_call_policies.py
- type_caster<void>
- ergebnisse/README.md
- Basismodell-Referenz-Eval — Ergebnis (Pipeline-Validierung)
- W&B-Auswertung — Run `g1_dex3_blockstacking_v1`
- setup_and_train_Container-build.sh
- internals.h
- EigenConformable
- Portabilität — das Repo auf einem fremden Rechner betreiben
- __main__.py
- Pybind11Extension
- C
- dict
- PartialStruct
- test_numpy_vectorize.py
- v3.8.6
- Dokumentation — Übersicht
- blend
- g1_gripper_blockstack_env.py
- TEST_SUBMODULE
- .load
- ParallelCompile
- Widget
- test_modules.py
- SimpleStruct
- Bewertung der Sim-Umsetzung — Sinnvoll & korrekt? (2026-06-05)
- GPU-Eignung für die Isaac-Sim-Closed-Loop-Sim auf GWDG
- Optionale Trainings-Features (getrennt schaltbar)
- Fixes aus dem ersten Trainingsdurchlauf
- extract_block_layout.py
- g1_gripper_cfg.py
- update_sim_image.sh
- entrypoint.sh
- setup_and_train_DockerHub-pull.sh
- update_image.sh
- .init
- _Handler
- Usages
- body.cpp
- functional.h
- argument_record
- .load
- DerivedWidget
- DtypeSizeCheck
- Repository Guidelines
- Baseline-Closed-Loop-Test — stock Unitree G1 + Dex1-Greifer (UNITREE_G1)
- W&B Offline-Sync auf KISSKI
- entrypoint_rl.sh
- run_trajectory
- phase_b_test.py
- list
- typeid.h
- array_info<std::array<T, N>>
- test_async.py
- DtypeCheck
- parametrize
- Läufe 33/34 (`runs/20260814/03`, `runs/20260814/04`): der TUNE_VISUAL-Checkpoint im Closed Loop
- Schritt 2 — Domain-Gap neu messen (`server_rl_run.sh gap`)
- LaTeX — Projektarbeit (Ausarbeitung)
- Projektarbeit Humanoider Roboter
- void_caster
- check_action_norm.py
- b2_description/README.md
- b2w_description/README.md
- go2_description/README.md
- doc
- is_method
- module_local
- name
- scope
- sibling
- format_descriptor<T, detail::enable_if_t<std::is_arithmetic<T>::value>>
- size_in_ptrs
- operator()
- setup.py
- MyBase
- PYBIND11_EMBEDDED_MODULE
- AliasedHasOpNewDelSize
- EnumStruct
- SimpleStructReordered
- test_numpy_vectorize.cpp
- MsgSerializer
- split_own_args
- Unitree G1 Description (URDF & MJCF)
- Unitree H1_2 Description (URDF & MJCF)
- joint_controller.h
- setup
- select_indices_impl<index_sequence<IPrev...>, I, B, Bs...>
- get_shared_data
- pair
- TEST_SUBMODULE
- TEST_SUBMODULE
- GR00T-N1.6: ONNX/TensorRT und schnelleres Sim-Rendering
- Training — Fine-tuning von GR00T N1.6
- RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung
- kisski_replay_submit.sh
- kisski_robocasa_ref_submit.sh
- kisski_sim_submit.sh
- _enable_hf_transfer
- kisski_open_loop_eval.sh
- download_data.sh
- CustomOperatorNew
- control_via_keyboard.cpp
- bug-report.md
- process_attribute<is_new_style_constructor>
- A_Tpl
- arg
- process_attribute<kw_only>
- summarize_run
- builtin_exception
- exactly_one
- ProtectedB
- is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>>
- test_simple_setup_py
- xfail
- compare_domain_gap.sh
- recolor_hands_black.py
- test_modules.cpp
- b2_description_mujoco/README.md
- name_space
- ComplexStruct
- policy_latency.py
- process_attribute<is_final>
- arr
- test_greedy_string_overload
- check-style.sh
- wandb-run-charts.html.md
- copilot-instructions.md
- convert_urdf_to_usd.py
- kisski_rl_submit.sh
- kisski_submit.sh
- process_attribute<prepend>
- type_caster<Type, enable_if_t<is_eigen_other<Type>::value>>
- Weiterführende Arbeiten

## God Nodes (most connected - your core abstractions)
1. `handle` - 187 edges
2. `pybind11_fail()` - 60 edges
3. `function_record` - 48 edges
4. `array` - 47 edges
5. `ptr` - 45 edges
6. `G1Dex3BlockstackEnv` - 41 edges
7. `UnitreeJointController` - 41 edges
8. `error_already_set()` - 41 edges
9. `UDP` - 40 edges
10. `server_rl_run.sh script` - 35 edges

## Surprising Connections (you probably didn't know these)
- `run_trajectory()` --calls--> `slice`  [EXTRACTED]
  Simulation/scripts/finger_span_openloop.py → data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/python_wrapper/third-party/pybind11/include/pybind11/pytypes.h
- `main()` --references--> `LoopFunc`  [INFERRED]
  data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/example/example_joystick.cpp → data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/include/unitree_legged_sdk/loop.h
- `main()` --references--> `LoopFunc`  [INFERRED]
  data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/example/example_position.cpp → data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/include/unitree_legged_sdk/loop.h
- `main()` --references--> `LoopFunc`  [INFERRED]
  data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/example/example_velocity.cpp → data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/include/unitree_legged_sdk/loop.h
- `main()` --references--> `LoopFunc`  [INFERRED]
  data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/example/example_walk.cpp → data/unitree_ros/unitree_ros_to_real/unitree_legged_sdk/include/unitree_legged_sdk/loop.h

## Import Cycles
- None detected.

## Communities (342 total, 30 thin omitted)

### Community 0 - "handle"
Cohesion: 0.09
Nodes (27): load(), type_caster<bool>, type_caster<T, enable_if_t<std::is_arithmetic<T>::value && !is_std_char_type<T>::value>>, ensure(), args_proxy, delattr(), error_already_set(), generic_item (+19 more)

### Community 1 - "cast.h"
Cohesion: 0.06
Nodes (35): always_construct_holder, value, get_thread_state_unchecked(), false_type, PyThreadState, true_type, handle_type_name, handle_type_name<args> (+27 more)

### Community 2 - "cpp_function"
Cohesion: 0.06
Nodes (35): postcall(), precall(), add_class_method(), add_object(), all_type_info_get_cache(), cpp_function, enum_, enum_base (+27 more)

### Community 3 - "class.h"
Cohesion: 0.08
Nodes (47): add_patient(), clear_instance(), clear_patients(), deregister_instance(), deregister_instance_impl(), enable_buffer_protocol(), enable_dynamic_attributes(), get_fully_qualified_tp_name() (+39 more)

### Community 4 - "init.h"
Cohesion: 0.05
Nodes (46): Alias, Cpp, alias_constructor, construct(), construct_alias_from_cpp(), construct_or_initialize(), constructor, execute() (+38 more)

### Community 5 - "pybind11.h"
Cohesion: 0.06
Nodes (41): any_of<is_holder<T>, is_subtype<T>, is_base<T>>, add_base(), call_operator_delete(), create_extension_module(), def_submodule(), get_overload(), get_override(), get_type_overload() (+33 more)

### Community 6 - "server_rl_run.sh"
Cohesion: 0.14
Nodes (46): livestream_active(), livestream_app_flags(), livestream_banner(), livestream_init(), livestream_video_dir(), log(), lib_livestream.sh script, warn() (+38 more)

### Community 7 - "Sim-Eval Implementation Notes — Lessons Learned & aktueller Stand"
Cohesion: 0.04
Nodes (46): 10. Kamera-Rekonstruktion aus dem Dataset, 11. Diagnose: Open-Loop-Replay — Sim führt Aktionen korrekt aus, 12. Aktueller Stand (2026-06-04), 13. Physics Calibration Session (2026-06-04), 14.1 Finger-Aktuatoren (`g1_dex3_cfg.py`), 14.2 Sign-Convention-Fix für proximale Fingergelenke, 14.3 Würfel-Reibung (`g1_dex3_blockstack_env.py`), 14.4 Tischhöhe und Würfelposition (+38 more)

### Community 8 - "Anleitung — GR00T N1.6 Fine-tuning"
Cohesion: 0.04
Nodes (45): 1. `docker cp` auf den Host, 2. Aus dem Container heraus hochladen (HuggingFace), 3. Auf vast.ai per SSH, 4. KISSKI: rsync vom Cluster, A1. vast.ai-Konto vorbereiten, A2. Instanz auswählen, A3. Image und Env-Vars eintragen, A4. Launch (+37 more)

### Community 9 - "GR00T N1.6 — Unitree G1 + DEX3 Setup-Dokumentation"
Cohesion: 0.05
Nodes (44): 1. Container-Analyse, 2. GR00T N1.6 installieren, 2-Kamera-Datensätze, 3. DEX3-Kompatibilität prüfen, 4-Kamera-Datensätze, 4. Modality-Config erstellen, 5. Datensatz auf HuggingFace suchen, 6. Datensatz vorbereiten (+36 more)

### Community 10 - "object"
Cohesion: 0.10
Nodes (13): PyObject, object_api<Derived>::operator()(), array_t(), bool_, borrowed_t, buffer, ellipsis, Name() (+5 more)

### Community 11 - "render_cotrain_dataset.py"
Cohesion: 0.10
Nodes (43): block_positions(), consistency_report(), episode_paths(), episode_window(), expected_length(), finalize_meta(), find_grasp_points(), hand_spreads_and_centroids() (+35 more)

### Community 12 - "class_"
Cohesion: 0.08
Nodes (40): execute(), execute_cast(), Class, op_, op_impl, __self(), self_t, undefined_t (+32 more)

### Community 13 - "test_eigen.py"
Cohesion: 0.06
Nodes (28): array_copy_but_one(), assert_equal_ref(), assert_keeps_alive(), assert_sparse_equal_ref(), assign_both(), Eigen doesn't support (as of yet) negative strides. When a function takes an…, Tests various ways of returning references and non-referencing copies, Tests Eigen's ability to mutate numpy values (+20 more)

### Community 14 - "Closed-Loop-Sim auf vast.ai — Schritt-für-Schritt-Anleitung"
Cohesion: 0.05
Nodes (39): 3a) URDF beschaffen, 3b) USD erzeugen, 3c) USD aufbewahren, 4a) GPU auswählen, 4b) Instance Configuration, `CHECKPOINT_PATH leer oder existiert nicht`, Closed-Loop-Sim auf vast.ai — Schritt-für-Schritt-Anleitung, `createDLSSContext error` / Rendering-Fehler (+31 more)

### Community 15 - "G1Dex3BlockstackEnv"
Cohesion: 0.08
Nodes (21): G1Dex3BlockstackEnv, Any, DirectRLEnv, Tensor, Wendet das in _pre_physics_step berechnete Target an (render_interval-mal)., Reward je nach cfg.reward_mode. "binary" (Default): spärlicher 0/1-Success-…, (num_envs, 2, 3) Weltpositionen der beiden Handwurzel-Links (links, rechts)., Weltpositionen der Kontaktflächen (Fingerkuppen), (num_envs, n, 3). Gemeinsame… (+13 more)

### Community 16 - "array"
Cohesion: 0.12
Nodes (17): buffer, array, array_proxy(), broadcast(), byte_offset(), check_dimensions(), check_flags(), data() (+9 more)

### Community 17 - "operator<"
Cohesion: 0.06
Nodes (17): accessor, dict_readonly, key, obj, pos, value, generic_iterator, It (+9 more)

### Community 18 - "sequence"
Cohesion: 0.09
Nodes (9): sequence, set, tuple, item_accessor, list_accessor, sequence_accessor, sequence_iterator, tuple_accessor (+1 more)

### Community 19 - "doc"
Cohesion: 0.07
Nodes (23): doc(), Sanitize docstrings and add custom failure explanation, test_docstrings(), test_sparse_signature(), parametrize, skipif, xfail, C++ default and converting constructors are equivalent to type calls in Python (+15 more)

### Community 20 - "convert.h"
Cohesion: 0.13
Nodes (23): BmsCmd, BmsState, Cartesian, MotorCmd, MotorState, ConstPtr, HighCmd, HighState (+15 more)

### Community 21 - "numpy.h"
Cohesion: 0.07
Nodes (29): array_info, array_info<char[N]>, array_info_scalar, extents, is_array, is_empty, array_info<std::array<char, N>>, array_info<T[N]> (+21 more)

### Community 22 - "test_tagbased_polymorphic.cpp"
Cohesion: 0.11
Nodes (22): Animal, kind, name, name_of_kind, type_of_kind, Cat, Chihuahua, itype (+14 more)

### Community 23 - "GR00T N1.6 Fine-tuning — Unitree G1 + DEX3 Hand"
Cohesion: 0.06
Nodes (33): 1.1 Modellgewichte herunterladen, 1.2 Datensatz bereitstellen, 1.3 Ausgabeverzeichnis anlegen, 1.4 Umgebung aktivieren, 1. Voraussetzungen, 2.1 Minimaler Start (1 GPU, ohne W&B), 2.2 Empfohlener Start (mit allen relevanten Parametern), 2.3 Parameter-Erklärung (DEX3-spezifisch) (+25 more)

### Community 24 - "UDP"
Cohesion: 0.06
Nodes (33): UDP, accessible, blockTimeout, connected, GetRecv, init, InitCmdData, initiativeDisconnect (+25 more)

### Community 25 - "constructor_stats.h"
Cohesion: 0.15
Nodes (24): ConstructorStats, copy_assignments, copy_constructions, default_constructions, _instances, move_assignments, move_constructions, format_ptrs() (+16 more)

### Community 26 - "msg"
Cohesion: 0.07
Nodes (38): msg(), Sanitize messages and add custom failure explanation, test_inheritance(), test_inheritance_init(), test_instance(), test_reentrant_implicit_conversion_failure(), create_and_destroy(), skipif (+30 more)

### Community 27 - "test_class.py"
Cohesion: 0.06
Nodes (19): xfail, #511: problem with inheritance + overwritten def_static, Ensure the lifetime of temporary objects created for implicit conversions, Tests that class-specific operator new/delete functions are invoked, Expose protected member functions to Python using a helper class, Tests that simple POD classes can be constructed using C++11 brace…, Instances must correctly increase/decrease the reference count of their types…, Tests that a properly qualified name is set in __qualname__ (even in pre-3.3,… (+11 more)

### Community 28 - "LiveView"
Cohesion: 0.09
Nodes (11): LiveView, Frame-Slot + HTTP-Server. Haelt immer nur das JEWEILS LETZTE Bild. Kein Puffer,…, Zaehlt den Aufruf und sagt, ob dieser Frame dran ist (LIVE_VIEW_EVERY_N)., Frame(s) veroeffentlichen. `frames`: Array/Tensor oder {name: Array/Tensor}., Bequemer Weg aus dem Env-Obs-Dict: nimmt video.<cam> fuer env_index. Ist keine…, Metriken aktualisieren ohne neues Bild (z. B. am Iterations-Ende)., Blockiert bis ein Frame neuer als last_seq da ist. -> (bytes|None, seq)., Torch-Tensor oder numpy-Array -> (H, W, 3) uint8 numpy auf der CPU. Akzeptiert… (+3 more)

### Community 29 - "UnitreeJointController"
Cohesion: 0.06
Nodes (31): Controller<hardware_interface::EffortJointInterface>, ServoCmd, Subscriber, UnitreeJointController, command, controller_state_publisher_, init, isCalf (+23 more)

### Community 30 - "T"
Cohesion: 0.11
Nodes (17): m, ref, shared_ptr, T, unique_ptr, custom_unique_ptr, impl, holder_helper<ref<T>> (+9 more)

### Community 31 - "vector"
Cohesion: 0.12
Nodes (10): broadcast_trivial, string, vector, Type, vectorize_returned_array, vectorize_returned_array<Func, void, Args...>, initializer_list, T (+2 more)

### Community 32 - "multiThread"
Cohesion: 0.10
Nodes (12): Imu, MotorState, NodeHandle, string, Subscriber, WrenchStamped, multiThread, footForce_sub (+4 more)

### Community 33 - "type_info"
Cohesion: 0.13
Nodes (23): all_type_info_populate(), name, detail::type_info *get_type_info(const std::type_index &tp,
                                                          bool throw_if_missing = false)(), detail::type_info* get_type_info(PyTypeObject *type)(), get_global_type_info(), get_local_type_info(), get_type_handle(), instance (+15 more)

### Community 34 - "rl_finetune.py"
Cohesion: 0.11
Nodes (29): _action_spec(), _assemble_phys_action(), _build_action_mask(), build_value_head(), compute_gae(), _env_flag(), _env_int(), _flat_to_nested() (+21 more)

### Community 35 - "type_caster_base"
Cohesion: 0.21
Nodes (7): Constructor, itype, polymorphic_type_hook, polymorphic_type_hook_base, polymorphic_type_hook_base<itype, detail::enable_if_t<std::is_polymorphic<itype>::value>>, type_caster_base, name

### Community 36 - "test_builtin_casters.py"
Cohesion: 0.07
Nodes (23): skipif, Tests the ability to pass bytes to C++ string-accepting functions. Note that…, Tests support for C++17 string_view arguments and return values, Tests unicode conversion and error reporting., Issue #929 - out-of-range integer values shouldn't be accepted, std::pair <-> tuple & std::tuple <-> tuple, Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None, None passed as various argument types should defer to other overloads (+15 more)

### Community 37 - "PolicyClient"
Cohesion: 0.08
Nodes (21): PolicyClient, build_obs(), MsgSerializer, PolicyClient, ndarray, Vendored GR00T PolicyClient für den Isaac-Lab-Sim-Container. Enthält nur die…, Setzt den Server-internen Policy-State zurück (neue Episode)., Sendet eine Observation (im Gr00tSimPolicyWrapper-Format) und empfängt Action-… (+13 more)

### Community 38 - "load_impl"
Cohesion: 0.11
Nodes (12): copyable_holder_caster, holder, shared_ptr, load_impl(), type_caster<std::shared_ptr<T>>, value_and_holder, index, inst (+4 more)

### Community 39 - "test_copy_move.cpp"
Cohesion: 0.10
Nodes (19): copy_move_policies, CopyOnlyInt, value, m, return_value_policy, empty, instance_, lacking_copy_ctor (+11 more)

### Community 40 - "function_record"
Cohesion: 0.08
Nodes (26): function_record, args, data, doc, has_args, has_kw_only_args, has_kwargs, is_constructor (+18 more)

### Community 41 - "test_stl.py"
Cohesion: 0.08
Nodes (22): skipif, Properties use the `reference_internal` policy by default. If the underlying…, #171: Can't return reference wrappers (or STL structures containing them), Trying convert `list` to a `std::vector`, or vice versa, without including…, Check if a string is NOT implicitly converted to a list, which was the behavior…, check fix for issue #1561, std::valarray <-> list, Tests that stl casters preserve lvalue/rvalue context for container values (+14 more)

### Community 42 - "return_value_policy"
Cohesion: 0.11
Nodes (15): caster_t, cast(), cast_impl(), get_object_handle(), holder_type, return_value_policy, U, object_or_cast() (+7 more)

### Community 43 - "detail/common.h"
Cohesion: 0.06
Nodes (36): bool_constant, bools, constexpr_last(), deferred_type, exactly_one<P, Default>, format_descriptor, false_type, intrinsic_type (+28 more)

### Community 44 - "OstreamRedirect"
Cohesion: 0.11
Nodes (19): add_ostream_redirect(), class_, unique_ptr, OstreamRedirect, do_stderr_, do_stdout_, redirect_stderr, redirect_stdout (+11 more)

### Community 45 - "dtype"
Cohesion: 0.08
Nodes (20): array_descriptor_proxy(), compare_buffer_info<T, detail::enable_if_t<detail::is_pod_struct<T>::value>>, dtype, field_descriptor, descr, format, name, offset (+12 more)

### Community 46 - "string"
Cohesion: 0.10
Nodes (21): check_dimensions_impl(), format_descriptor<char[N]>, format_descriptor<std::array<char, N>>, format_descriptor<T, detail::enable_if_t<detail::array_info<T>::is_array>>, format_descriptor<T, detail::enable_if_t<detail::is_pod_struct<T>::value>>, format_descriptor<T, detail::enable_if_t<std::is_enum<T>::value>>, get_type_info(), PYBIND11_NOINLINE (+13 more)

### Community 47 - "PyArrayDescr_Proxy"
Cohesion: 0.07
Nodes (27): PyObject_HEAD, PyArray_Proxy, base, data, descr, dimensions, flags, nd (+19 more)

### Community 48 - "test_methods_and_attributes.py"
Cohesion: 0.08
Nodes (21): parametrize, xfail, Static property getter and setters expect the type object as the their only…, Overriding pybind11's default metaclass changes the behavior of…, When returning an rvalue, the return value policy is automatically changed from…, #283: __str__ called on uninitialized instance when constructor arguments…, Tests that explicit lvalue ref-qualified methods can be called just like their…, Check to see if the normal overload order (first defined) and prepend overload… (+13 more)

### Community 49 - "test_numpy_dtypes.py"
Cohesion: 0.11
Nodes (15): assert_equal(), dt_fmt(), packed_dtype(), packed_dtype_fmt(), partial_dtype_fmt(), partial_ld_offset(), partial_nested_fmt(), fixture (+7 more)

### Community 50 - "lib_resume_guard.sh"
Cohesion: 0.14
Nodes (21): err(), log(), resume_guard(), lib_resume_guard.sh script, warn(), err(), log(), lib_split.sh script (+13 more)

### Community 51 - "attr.h"
Cohesion: 0.10
Nodes (23): arithmetic, buffer_protocol, call_guard, call_guard<T>, call_guard<T, Ts...>, dynamic_attr, function_call::function_call(), is_operator (+15 more)

### Community 52 - "buffer_info"
Cohesion: 0.12
Nodes (19): buffer_info, format, itemsize, m_view, ndim, ownview, ptr, readonly (+11 more)

### Community 53 - "G1Dex3BlockstackEnvCfg"
Cohesion: 0.10
Nodes (21): CameraCfg, main(), Dump-Skript: speichert je ein RGB-Standbild der vier POLICY-Kameras der Isaac-…, G1Dex3BlockstackEnvCfg, G1Dex3BlockstackSceneCfg, _make_sim_cfg(), _plain_camera_cfg(), configclass (+13 more)

### Community 54 - "iostream"
Cohesion: 0.28
Nodes (6): m, string, noisy_funct_dual(), noisy_function(), TEST_SUBMODULE(), iostream

### Community 55 - "UnitreeDrawForcePlugin"
Cohesion: 0.09
Nodes (20): ConnectionPtr, ElementPtr, NodeHandle, string, Subscriber, WrenchStamped, UnitreeDrawForcePlugin, force_sub (+12 more)

### Community 56 - "TEST_SUBMODULE"
Cohesion: 0.13
Nodes (15): m, string, MyException, MyException2, message, MyException3, message, MyException4 (+7 more)

### Community 57 - "Auswertung — erster vollständiger Trainingsdurchlauf (`g1_dex3_blockstacking_v1`)"
Cohesion: 0.08
Nodes (25): 1. Kurzfazit (TL;DR), 2. Lauf-Eckdaten (abgeschlossen), 3. Trainingsdynamik — gesund ✅, 4.1 Closed-Loop-Sim-Eval (Modell steuert die Sim), 4.2 Open-Loop-Dataset-Replay (echte Aktionen, KEIN Modell), 4.3 Open-Loop-Modell-Eval (echte Dataset-Beobachtungen → predicted vs. GT-Actions), 4. Verhaltens-Evaluation — drei Diagnosen, 5. Synthese & Diagnose (+17 more)

### Community 58 - "UnitreeFootContactPlugin"
Cohesion: 0.09
Nodes (20): ContactSensorPtr, ConnectionPtr, ElementPtr, NodeHandle, Publisher, string, WrenchStamped, UnitreeFootContactPlugin (+12 more)

### Community 59 - "test_custom_type_casters.cpp"
Cohesion: 0.13
Nodes (15): custom_type_casters, ArgAlwaysConverts, ArgInspector1, arg, ArgInspector2, arg, m, return_value_policy (+7 more)

### Community 60 - "Custom"
Cohesion: 0.10
Nodes (22): LowCmd, LowState, Custom, cmd, dt, Kd, Kp, motiontime (+14 more)

### Community 61 - "str"
Cohesion: 0.16
Nodes (12): eval(), eval_file(), exec(), bytes, format(), string, release(), str (+4 more)

### Community 62 - "print_destroyed"
Cohesion: 0.24
Nodes (9): print_copy_created(), print_created(), print_default_created(), print_destroyed(), print_move_created(), NoConstructor, PYBIND11_OVERRIDE(), PyTF6() (+1 more)

### Community 63 - "convert_v3_to_v2_standalone.py"
Cohesion: 0.23
Nodes (22): convert_data(), convert_dataset(), convert_episodes_metadata(), convert_info(), convert_tasks(), convert_videos(), copy_global_stats(), _extract_video_segment() (+14 more)

### Community 64 - "unitree_legged_sdk.h"
Cohesion: 0.12
Nodes (15): LowCmd, LowState, Custom, cmd, dt, _keyData, motiontime, RobotControl (+7 more)

### Community 65 - "type_record"
Cohesion: 0.10
Nodes (21): PYBIND11_NOINLINE, type_info, metaclass(), process_attribute<metaclass>, type_record, bases, buffer_protocol, default_holder (+13 more)

### Community 66 - "unpacking_collector"
Cohesion: 0.13
Nodes (14): collect_arguments(), policy, string, Tuple, make_tuple(), simple_collector, m_args, tuple_caster (+6 more)

### Community 67 - "conftest.py"
Cohesion: 0.12
Nodes (17): gc_collect(), _make_explanation(), pytest_assertrepr_compare(), pytest_configure(), Hook to insert custom failure explanation, Suppress the desired exception, Run the garbage collector twice (needed when running reference counting tests…, For triple-quote strings (+9 more)

### Community 68 - "Implementierungsplan — Live-Ansicht der Isaac-Lab-Sim"
Cohesion: 0.09
Nodes (23): 0. Was sich seit v1 geändert hat, 1. Zwei Spuren — und welche wofür, 2. Befund am bestehenden Code — 6 konkrete Defekte, 3.1 Funktionsweise, Ports, Clients (Isaac Sim 6.0), 3.2 Netzkonfiguration auf `ikr-ki-server-01`, 3.3 Code-Änderungen, 3.4 vast.ai-Sonderfall, 3. Spur A — WebRTC-Viewport (+15 more)

### Community 69 - "pybind11_tests.h"
Cohesion: 0.06
Nodes (28): async_module, bind_local(), class_, T, LocalBase, i, MixGL, MixGL2 (+20 more)

### Community 70 - "HasOpNewDel"
Cohesion: 0.16
Nodes (7): HasOpNewDel, i, HasOpNewDelBoth, i, HasOpNewDelSize, i, uint32_t

### Community 71 - "pybind11_tests"
Cohesion: 0.07
Nodes (9): Tests that returning a pointer to a type that gets converted with a custom type…, test_custom_caster_destruction(), test_noconvert_args(), test_pointers(), #393: need to return NotSupported to ensure correct arithmetic operator behavior, #328: first member in a class can't be used in operators, test_nested(), test_operators_notimplemented() (+1 more)

### Community 72 - "ExampleMandA"
Cohesion: 0.10
Nodes (3): string, ExampleMandA, value

### Community 73 - "test_multiple_inheritance.cpp"
Cohesion: 0.13
Nodes (15): BaseN, i, m, string, TEST_SUBMODULE(), Vanilla, VanillaStaticMix1, static_value (+7 more)

### Community 75 - "3. Ablauf"
Cohesion: 0.09
Nodes (23): 1. Warum, 2. Was dafür gebaut wurde, 3.0 Zuerst die Würfellage aus den Realbildern — `layout` (seit 2026-08-17), 3.1 Rauchtest zuerst (≈ 2 min) — prüft die Klempnerei, nicht die Inhalte, 3.1b Dann der Scan über den ganzen Satz — das ist die inhaltliche Prüfung, 3.2 Der lange Lauf, 3.2a Nur bis zum Griff rendern — `RENDER_STOP_AT_GRASP` (Default an, seit 2026-08-17), 3.2b Die QA-Zahl: Bild-Aktions-Konsistenz, nicht Aufgabenerfolg (+15 more)

### Community 76 - "type_caster_generic"
Cohesion: 0.22
Nodes (5): type_caster_generic, cpptype, type_caster_generic, typeinfo, value

### Community 77 - "env.py"
Cohesion: 0.10
Nodes (6): xfail, test_eval_file(), parametrize, xfail, test_roundtrip(), test_roundtrip_with_dict()

### Community 78 - "Pet"
Cohesion: 0.67
Nodes (3): string, Pet, name_

### Community 79 - "test_class.cpp"
Cohesion: 0.14
Nodes (15): def, BaseClass, BreaksBase, BreaksTramp, DerivedClass1, DerivedClass2, foo(), Invalid (+7 more)

### Community 80 - "test_factory_constructors.cpp"
Cohesion: 0.12
Nodes (15): string, TestFactory1, value, TestFactory2, value, TestFactory3, value, TestFactory4 (+7 more)

### Community 81 - "test_gil_scoped.py"
Cohesion: 0.14
Nodes (20): _python_to_cpp_to_python(), _python_to_cpp_to_python_from_threads(), Calls different C++ functions that come back to Python., Calls different C++ functions that come back to Python, from Python threads., # TODO: FIXME, sometimes returns -11 (segfault) instead of 0 on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread. It runs in a…, # TODO: FIXME on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread multiple times in… (+12 more)

### Community 82 - "Lokomotion des Unitree G1 freischalten — Recherche"
Cohesion: 0.10
Nodes (21): 0. TL;DR — Die zentrale Erkenntnis, 1.1 Der Sim-Root-Link ist festgeschweißt, 1.2 Der Action-/State-Vektor hat keine Beine, 1.3 Die Beine sind aber im Asset vorhanden, 1.4 Der reale Datensatz enthält keine Lokomotion, 1. Warum der Roboter aktuell fixiert ist (Code-Analyse), 2. Wie der Unitree G1 mechanisch aufgebaut ist, 3.1 Entkoppelte Architektur (Decoupled WBC) (+13 more)

### Community 83 - "server_robocasa_ref_run.sh"
Cohesion: 0.30
Nodes (19): build_eval_env(), collect_videos(), do_clean(), do_eval(), do_fix_flash_attn(), do_preflight(), do_setup(), do_shell() (+11 more)

### Community 84 - "Reinforcement-Learning-Training — Recherche & Umsetzungsplan"
Cohesion: 0.10
Nodes (20): 1. Warum überhaupt RL? — Motivation aus dem ersten Lauf, 2. Zwei grundverschiedene „RL"-Ebenen — nicht verwechseln, 3.1 Ein Environment mit `reset()` / `step()`, 3.2 Eine **Reward-Funktion**, 3.3 Ein RL-Algorithmus, der zur **Flow-Matching-Policy passt**, 3.4 Rollout-Infrastruktur (Policy ↔ Env-Kommunikation), 3.5 Rechenleistung — und ein projektspezifischer GPU-Konflikt ⚠️, 3. Was ist technisch nötig? — Die Bausteine (+12 more)

### Community 85 - "stl"
Cohesion: 0.13
Nodes (12): buffers, m, TEST_SUBMODULE(), m, TEST_SUBMODULE(), m, TEST_SUBMODULE(), m (+4 more)

### Community 86 - "complex.h"
Cohesion: 0.11
Nodes (14): builtin_casters, format_descriptor<std::complex<T>, detail::enable_if_t<std::is_floating_point<T>::value>>, c, value, complex, return_value_policy, string, T (+6 more)

### Community 87 - "teleForceCmd"
Cohesion: 0.16
Nodes (16): NodeHandle, Publisher, main(), quit(), teleForceCmd, Force, force_pub, Fx (+8 more)

### Community 88 - "internals"
Cohesion: 0.11
Nodes (19): instance, unordered_set, internals, default_metaclass, direct_conversions, inactive_override_cache, instance_base, loader_patient_stack (+11 more)

### Community 89 - "EigenProps"
Cohesion: 0.12
Nodes (19): eigen_array_cast(), EigenProps, cols, descriptor, dynamic, dynamic_stride, fixed, fixed_cols (+11 more)

### Community 90 - "test_local_bindings.py"
Cohesion: 0.11
Nodes (15): xfail, Load a `py::module_local` type that's only registered in an external module, Local types take precedence over globally registered types: a module with a…, Makes sure the internal local type map differs across the two modules, One module uses a generic vector caster from `<pybind11/stl.h>` while the other…, Tests that duplicate `py::module_local` class bindings work across modules, Tests that attempting to register a non-local type in multiple modules fails, Tests expected failure when registering a class twice with py::local in the… (+7 more)

### Community 91 - "test_multiple_inheritance.py"
Cohesion: 0.12
Nodes (14): skipif, xfail, Mixing bases with and without static properties should be possible and the…, Mixing bases with and without dynamic attribute support, Returning an offset (non-first MI) base class pointer should recognize the…, Tests returning an offset (non-first MI) base class pointer to a derived…, Tests that diamond inheritance works as expected (issue #959), test_diamond_inheritance() (+6 more)

### Community 92 - "test_virtual_functions.cpp"
Cohesion: 0.14
Nodes (14): Base, m, string, DispatchIssue, ExampleVirt, pure_virtual, run_bool, state (+6 more)

### Community 93 - "Auswertung — dritter Trainingsdurchlauf, Vision-Encoder **mit** Split & Augmentierung (`g1_dex3_blockstacking_vision_v2`)"
Cohesion: 0.11
Nodes (19): 10. Verwendete Artefakte / Quellen, 1. Kurzfazit (TL;DR), 2. Lauf-Eckdaten, 3. Trainingsdynamik — gesund ✅, 4.1 Overfitting ist erstmals sichtbar, 4.2 Was das rückwirkend über Lauf 1 und 2 sagt, 4.3 Die Verschlechterung sind Ausreißer, keine Breitendegradation, 4.4 Episode 288 lernt ab Step 10.000 nichts mehr (+11 more)

### Community 94 - "G1GripperBlockstackEnv"
Cohesion: 0.16
Nodes (8): G1GripperBlockstackEnv, Any, DirectRLEnv, ndarray, Tensor, Closed-Loop-Env für die GR00T-N1.6-Baseline (stock G1 + Dex1-Greifer)., actions: (N, 16) = [left_arm(7), right_arm(7), left_hand(1), right_hand(1)].…, Numpy-Obs (Env 0) für den Eval-Runner / client_g1.build_obs.

### Community 96 - "Loop"
Cohesion: 0.12
Nodes (15): Callback, main(), main(), string, Loop, _bind_cpu_flag, _bindCPU, entryFunc (+7 more)

### Community 97 - "chrono.h"
Cohesion: 0.19
Nodes (13): Clock, _period, duration_caster, get_duration(), duration, return_value_policy, type, type_caster<std::chrono::duration<Rep, Period>> (+5 more)

### Community 98 - "test_stl_binders.cpp"
Cohesion: 0.13
Nodes (14): unordered_map, Container, m, Map, E_nc, value, El, a (+6 more)

### Community 99 - "npy_api"
Cohesion: 0.22
Nodes (6): check_(), PyTypeObject, npy_api, PyArray_Type_, PyArrayDescr_Type_, PyVoidArrType_Type_

### Community 100 - "UserType"
Cohesion: 0.12
Nodes (10): IncType, UserType, i, ConvertibleFromUserType, i, TestPropRVP, sv1, sv2 (+2 more)

### Community 101 - "test_eigen.cpp"
Cohesion: 0.26
Nodes (11): adjust_matrix(), M, Ref, get_elem(), reset_ref(), reset_refs(), TEST_SUBMODULE(), x (+3 more)

### Community 102 - "Journey Into projektarbeit_humanoider_roboter"
Cohesion: 0.11
Nodes (17): 10. Lessons and Meta-Observations, 1. Project Genesis, 2. Architectural Evolution, 3. Key Breakthroughs, 4. Work Patterns, 5. Technical Debt, 6. Challenges and Debugging Sagas, 7. Memory and Continuity (+9 more)

### Community 103 - "dump_camera_poses.py"
Cohesion: 0.17
Nodes (17): camera_tangents(), first_attr(), frustum_report(), main(), quat_rotate(), Die USD-Stage direkt befragen — der einzige Zeuge, der nicht aus `cam.data`…, Halbwinkel-Tangenten aus der Intrinsik, die Isaac Lab SELBST meldet. -> (tan_h,…, Projiziert Szenenpunkte in eine Kamera. -> Zeilen (name, u, v, dist, drin?).… (+9 more)

### Community 104 - "test_numpy_array.cpp"
Cohesion: 0.23
Nodes (16): arr, arr_t, at_t(), auxiliaries(), T, T2, data(), data_t() (+8 more)

### Community 105 - "Capture"
Cohesion: 0.13
Nodes (7): Capture, Output, fixture, Extended `capsys` with context manager and custom equality operators, Basic output post-processing and comparison, Custom comparison for output without strict line ordering, Unordered

### Community 106 - "test_numpy_dtypes.cpp"
Cohesion: 0.11
Nodes (18): ArrayStruct, a, b, c, d, S, create_recarray(), mkarray_via_buffer() (+10 more)

### Community 107 - "Domain-Gap-Analyse: Real → Isaac Sim"
Cohesion: 0.12
Nodes (17): Baseline-Vergleiche, Der Blocker: cam_left_wrist, Domain-Gap-Analyse: Real → Isaac Sim, Empfohlene Reihenfolge, Ergebnisse, Fragestellung, Gesamtbild, Handlungsoptionen (+9 more)

### Community 108 - "Umgebungsanalyse: Training & Simulation"
Cohesion: 0.12
Nodes (17): 1. Frozen Vision Encoder + Sim-Eval — strukturell inkompatibel, 2. Training ohne Validation = Blindflug, 3. DEX3-Fingerqualität als bekanntes, ungelöstes Problem, Domain-Gap-Messung (2026-06-04) — ⚠️ überholt, Empfehlungen, Episode-Length-Diskrepanz (behoben), Fundamentale Denkfehler, Joint Sign-Flip — Convention-Leak-Risiko (+9 more)

### Community 109 - "PYBIND11_OVERRIDE"
Cohesion: 0.17
Nodes (10): B_Repeat, C_Repeat, get_noncopyable(), NCVirtTrampoline, PYBIND11_OVERRIDE(), unlucky_number(), get_string2, Movable (+2 more)

### Community 110 - "simulation/README.md"
Cohesion: 0.33
Nodes (4): Archiv, Einstieg, GPU-Anforderung (Kurzfassung), Simulation — Closed-Loop-Eval in Isaac Lab

### Community 111 - "test_constants_and_functions.cpp"
Cohesion: 0.16
Nodes (11): constants_and_functions, m, string, print_bytes(), return_bytes(), test_function1(), test_function2(), test_function3() (+3 more)

### Community 112 - "Custom"
Cohesion: 0.12
Nodes (11): HighCmd, HighState, LowCmd, LowState, Custom, high_cmd, high_state, high_udp (+3 more)

### Community 113 - "Custom"
Cohesion: 0.12
Nodes (11): HighCmd, HighState, LowCmd, LowState, Custom, high_cmd, high_state, high_udp (+3 more)

### Community 114 - "Custom"
Cohesion: 0.13
Nodes (15): main(), LowCmd, LowState, Custom, cmd, dt, motiontime, RobotControl (+7 more)

### Community 115 - "index_sequence"
Cohesion: 0.11
Nodes (28): index_sequence, concat(), descr, text, type_info, int_to_str, int_to_str<0, Digits...>, digits (+20 more)

### Community 116 - "pytypes.h"
Cohesion: 0.06
Nodes (45): add(), append(), arg, arg_v, arrow_proxy, value, as_unsigned(), cast() (+37 more)

### Community 117 - "pybind11_tests.cpp"
Cohesion: 0.29
Nodes (5): bind_ConstructorStats(), m, PYBIND11_MODULE(), test_initializer::test_initializer(), Initializer

### Community 118 - "stl.h"
Cohesion: 0.19
Nodes (17): call(), cast(), return_value_policy, T, Variant, operator()(), optional_caster, type_caster<std::experimental::optional<T>> (+9 more)

### Community 119 - "Pet"
Cohesion: 0.18
Nodes (8): Chimera, string, Dog, Hamster, Pet, m_name, m_species, Rabbit

### Community 120 - "test_interpreter.cpp"
Cohesion: 0.13
Nodes (10): "Execution frame", "Import error handling", "Pass classes and data between modules defined in C++ and Python", "Reload module from file", "Restart the interpreter", scope_exit, f_, "Subinterpreter" (+2 more)

### Community 121 - "test_methods_and_attributes.cpp"
Cohesion: 0.10
Nodes (16): shared_ptr, none1(), none2(), none3(), none4(), none5(), NoneTester, answer (+8 more)

### Community 122 - "test_operator_overloading.cpp"
Cohesion: 0.21
Nodes (11): abs(), C1, C2, m, string, hash<Vector2>, operator+(), TEST_SUBMODULE() (+3 more)

### Community 123 - "RoboCasa GR-1 Referenz-Eval — Bedienung"
Cohesion: 0.13
Nodes (13): Akzeptanzkriterien & Interpretation, Architektur (zwei Prozesse, eine Maschine), Artefakte (alle additiv, ändern nichts Bestehendes), Bekannte Klärungspunkte / Risiken, Env-Var-Referenz (alle Prefix `RC_`, kollisionsfrei), Ergebnis-JSON (`summary-<ts>.json`), Isolation — was bewusst NICHT verändert wird, Pfad A2 — Helferskript für generischen Docker-GPU-Server ★ empfohlen (+5 more)

### Community 124 - "Basismodell-Fähigkeiten & Referenzaufgabe zur Sim-Validierung"
Cohesion: 0.12
Nodes (16): 1.1 Modell, 1.2 Zero-shot oder Finetuning nötig?, 1.3 Mitgelieferte Embodiments (`embodiment_id.json` der Release-Checkpoint), 1.4 Closed-Loop-Benchmarks mit Erfolgsquoten, 1.5 Unitree G1 speziell, 1. Was ist GR00T N1.6 und was kann es bereits?, 2. Kritische Einordnung der Idee, 3. Implementierungsplan (+8 more)

### Community 125 - "Live-Ansicht — Isaac Sim auf dem eigenen Rechner öffnen"
Cohesion: 0.12
Nodes (16): Drei Live-Wege — welcher wofür, Env-Var-Referenz, Falls der Viewport schwarz bleibt, obwohl UDP offen ist, Fehlersuche, Für echte Hardware zählt eine andere Zahl, Live-Ansicht — Isaac Sim auf dem eigenen Rechner öffnen, Live **statt** Video (und wie man beides bekommt), Schritt 1 — Client auf dem Arbeitsrechner installieren (+8 more)

### Community 126 - "Schritt 3 — BC-Erfolgsrate in der Sim (`server_rl_run.sh eval`)"
Cohesion: 0.12
Nodes (16): Antwort (runs/20260808/26): der Griff scheitert auch ohne Modell, Der Test ohne Platzierungs-Annahme: `GRASP_MODE=hold`, Die eigentliche Messung (runs/20260808/25) — die Hand ist am Würfel, Die Würfel „springen" im Video — das ist der Test, kein Bug, Erste Messung (runs/20260808/24) — und warum sie so noch nichts entscheidet, Falscher Alarm „Sim-Eval ohne Erfolgsmarker beendet", Gate vor den 48 GPU-Stunden: drei Erklärungen, eine schon widerlegt, Gestufte Meilensteine in der Closed-Loop-Eval (2026-08-12) (+8 more)

### Community 127 - "Alloc"
Cohesion: 0.19
Nodes (13): Alloc, array_caster, Type, list_caster, reserve_maybe(), type_caster<std::array<Type, Size>>, type_caster<std::deque<Type, Alloc>>, type_caster<std::list<Type, Alloc>> (+5 more)

### Community 128 - "different_resolutions"
Cohesion: 0.14
Nodes (14): chrono, m, different_resolutions, timestamp_h, timestamp_m, timestamp_ms, timestamp_s, timestamp_us (+6 more)

### Community 129 - "CLAUDE.md"
Cohesion: 0.13
Nodes (14): Architecture, Build the image, Code style, Env-var configuration, graphify, Important: scripts are COPIED into the image, no host-side data persistence, Interactive shell (entrypoint bypassed when a command is passed), Key commands (+6 more)

### Community 130 - "multi_array_iterator"
Cohesion: 0.19
Nodes (10): common_iter, container_type, common_iterator, m_strides, p_ptr, multi_array_iterator, m_common_iterator, m_index (+2 more)

### Community 131 - "type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>"
Cohesion: 0.15
Nodes (16): Compare, map, set, unordered_map, unordered_set, Value, map_caster, set_caster (+8 more)

### Community 132 - "Custom"
Cohesion: 0.16
Nodes (13): HighCmd, HighState, Custom, cmd, dt, motiontime, RobotControl, safe (+5 more)

### Community 133 - "TEST_SUBMODULE"
Cohesion: 0.17
Nodes (9): m, TEST_SUBMODULE(), TestProperties, static_value, value, TestPropertiesOverride, static_value, value (+1 more)

### Community 134 - "function_call"
Cohesion: 0.09
Nodes (19): process_attributes, argument_loader, arg_names, argcasters, args_kwargs_are_last, args_pos, has_args, has_kwargs (+11 more)

### Community 135 - "type_info"
Cohesion: 0.13
Nodes (15): PyTypeObject, type_info, cpptype, default_holder, direct_conversions, get_buffer_data, holder_size_in_ptrs, implicit_casts (+7 more)

### Community 136 - "iterator"
Cohesion: 0.20
Nodes (12): Type, iterator_state, end, first_or_done, it, make_iterator(), make_key_iterator(), iterator (+4 more)

### Community 137 - "test_iostream.py"
Cohesion: 0.20
Nodes (8): redirect_stderr(), redirect_stdout(), test_err(), test_multi_captured(), test_not_captured(), test_redirect(), test_redirect_both(), test_redirect_err()

### Community 138 - "test_sequences_and_iterators.cpp"
Cohesion: 0.19
Nodes (12): A, m, pair, T, NonZeroIterator, ptr_, NonZeroSentinel, operator==() (+4 more)

### Community 139 - "test_sequences_and_iterators.py"
Cohesion: 0.16
Nodes (10): allclose(), isclose(), #2076: Exception raised by len(arg) should be propagated, #181: iterator passthrough did not compile, #388: Can't make iterators via make_iterator() with different r/v policies, Like math.isclose() from Python 3.5, test_iterator_passthrough(), test_iterator_rvp() (+2 more)

### Community 141 - "test_stl.cpp"
Cohesion: 0.15
Nodes (9): T, variant, hash<TplCtorClass>, OptionalHolder, member, TplCtorClass, type_caster<boost::variant<Ts...>>, visit_helper<boost::variant> (+1 more)

### Community 142 - "test_virtual_functions.py"
Cohesion: 0.13
Nodes (12): skipif, xfail, `A2`, unlike the above, is configured to always initialize the alias While the…, #159: virtual function dispatch has problems with similar-named functions, #392/397: overriding reference-returning functions, `A` only initializes its trampoline class when we inherit from it If we just…, test_alias_delay_initialization1(), test_alias_delay_initialization2() (+4 more)

### Community 143 - "Sim-Container auf KISSKI starten — Schritt-für-Schritt-Anleitung"
Cohesion: 0.13
Nodes (15): 5a) Einfacher Smoke-Test (Python läuft, kein Rendering), 5b) Kamera-Rendering-Test (Phase A, eigentliches Ziel), 5c) Kamera-Rendering-Test mit GUI (optional, nur im JupyterHPC-Desktop), Bekannte Probleme & Lösungen, Fallback: Ältere Isaac-Lab-Version, Nächste Phase nach erfolgreichem Phase-A+B-Test, Schritt 1 — SIF auf KISSKI ziehen (Login-Node, einmalig), Schritt 2 — JupyterHPC-Desktop starten (+7 more)

### Community 144 - "HPC-Training auf KISSKI"
Cohesion: 0.12
Nodes (16): Checkpoints nach dem Training sichern, HPC-Training auf KISSKI, Häufige KISSKI-Probleme, Job bleibt in Status `PD` (Pending), Job-Status verfolgen, `kisski_submit.sh` anpassen, `No space left on device` im Container, Pfad-Konfiguration — zwei Variablen, sonst nichts (+8 more)

### Community 145 - "Object"
Cohesion: 0.18
Nodes (8): atomic, T, Object, m_refCount, toString, ref, m_ptr, ref_tag

### Community 146 - "joint_controller.cpp"
Cohesion: 0.13
Nodes (12): NodeHandle, UnitreeJointController::effortLimits(), UnitreeJointController::init(), UnitreeJointController::positionLimits(), UnitreeJointController::setCommandCB(), UnitreeJointController::setTorqueCB(), UnitreeJointController::starting(), UnitreeJointController::velocityLimits() (+4 more)

### Community 147 - "Custom"
Cohesion: 0.18
Nodes (12): LowCmd, LowState, Custom, cmd, dt, motiontime, RobotControl, safe (+4 more)

### Community 148 - "Time"
Cohesion: 0.21
Nodes (4): m, PYBIND11_MODULE(), robot_interface, Time

### Community 149 - "type_caster<
    Eigen::Ref<PlainObjectType, 0, StrideType>,
    enable_if_t<is_eigen_dense_map<Eigen::Ref<PlainObjectType, 0, StrideType>>::value>
>"
Cohesion: 0.22
Nodes (9): Ref, unique_ptr, type_caster<
    Eigen::Ref<PlainObjectType, 0, StrideType>,
    enable_if_t<is_eigen_dense_map<Eigen::Ref<PlainObjectType, 0, StrideType>>::value>
>, copy_or_ref, need_writeable, ref, MapType, PlainObjectType (+1 more)

### Community 151 - "test_copy_move.py"
Cohesion: 0.14
Nodes (11): skipif, #389: rvp::move should fall-through to copy on non-movable objects, Cast some values in C++ via custom type casters and count the number of…, Call some functions that load arguments via custom type casters and count the…, Tests move/copy loads of std::optional arguments, An object with a private `operator new` cannot be returned by value, test_move_and_copy_casts(), test_move_and_copy_load_optional() (+3 more)

### Community 152 - "type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>>"
Cohesion: 0.13
Nodes (16): Allocator, basic_string, basic_string_view, CharT, load_bytes(), string_caster, UTF_N, type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>> (+8 more)

### Community 153 - "Development of pybind11"
Cohesion: 0.15
Nodes (12): 1. Building from the source directory, 2. Building from SDist, Build recipes, Clang-Tidy, Configuration options, Development of pybind11, Explanation of the SDist/wheel building design, Formatting (+4 more)

### Community 154 - "process_attribute_default"
Cohesion: 0.21
Nodes (8): base(), T, process_attribute<base<T>>, process_attribute<const char *>, process_attribute_default, process_attribute<pos_only>, process_attribute<T, enable_if_t<is_pyobject<T>::value>>, pos_only

### Community 155 - "PYBIND11_OVERRIDE_PURE"
Cohesion: 0.24
Nodes (6): gil_acquire(), PyModuleDef, PYBIND11_OVERRIDE_PURE(), VirtClass, pure_virtual_func, TEST_SUBMODULE

### Community 156 - "Auswertung — zweiter Trainingsdurchlauf mit Vision-Encoder (`g1_dex3_blockstacking_vision_v1`)"
Cohesion: 0.15
Nodes (13): 1. Kurzfazit (TL;DR), 2. Lauf-Eckdaten (abgeschlossen), 3. Trainingsdynamik — gesund ✅ (wie Lauf 1), 4. Verhaltens-Evaluation — Regression statt Verbesserung ❌, 5. Synthese & Diagnose, 6.1 `tune_visual = true` auf reinen Realdaten nicht weiterverfolgen, 6.2 Wenn der Vision-Pfad weiterverfolgt wird, 6.3 Eval endlich einschalten (gilt für beide Läufe) (+5 more)

### Community 157 - "Training"
Cohesion: 0.15
Nodes (13): Closed-Loop: Roboter bewegt sich kaum, greift nicht, `CUDA out of memory` (OOM), Daten / Checkpoints nach Job-Ende verschwunden, `flash-attn` / `--no-flash-attn`-Fehler auf älterer GPU, Job hängt lange in der Queue (`PENDING`), Kamera-Rendering scheitert: `createDLSSContext error`, KISSKI / HPC, Simulation (Isaac Lab / Isaac Sim) (+5 more)

### Community 158 - "Pfad A — vast.ai (Cloud-Miete)"
Cohesion: 0.15
Nodes (13): 4a) GPU auswählen, 4b) Instance Configuration, Live zusehen (`LIVE_VIEW=1`) — dringend empfohlen beim ersten großen Lauf, Pfad A — vast.ai (Cloud-Miete), Schritt 1 — Image bauen & pushen (einmalig), Schritt 2 — BC-Checkpoint als RL-Startpunkt bereitstellen, Schritt 3 — USD-Asset erzeugen (einmalig), Schritt 4 — Instanz auf vast.ai konfigurieren (+5 more)

### Community 159 - "run_robocasa_ref_eval.sh"
Cohesion: 0.26
Nodes (12): cleanup(), err(), JSON_ENTRIES, log(), MUJOCO_GL, ok(), PYOPENGL_PLATFORM, REF (+4 more)

### Community 160 - "entrypoint_baseline.sh"
Cohesion: 0.27
Nodes (10): cleanup(), err(), livestream_app_flags(), livestream_banner(), log(), ok(), PYTHONUNBUFFERED, entrypoint_baseline.sh script (+2 more)

### Community 161 - "entrypoint_sim.sh"
Cohesion: 0.27
Nodes (10): cleanup(), err(), livestream_app_flags(), livestream_banner(), log(), ok(), PYTHONUNBUFFERED, entrypoint_sim.sh script (+2 more)

### Community 162 - "measure_domain_gap.py"
Cohesion: 0.31
Nodes (12): cosine_distance(), crossview_mean(), discover_variants(), extract_embedding(), find_frame(), main(), ndarray, Path (+4 more)

### Community 163 - "is_copy_assignable<std::pair<T1, T2>>"
Cohesion: 0.19
Nodes (13): all_of, pair, T2, is_copy_assignable, is_copy_assignable<Container, enable_if_t<all_of<
        std::is_copy_assignable<Container>,
        std::is_same<typename Container::value_type &, typename Container::reference>
    >::value>>, is_copy_assignable<std::pair<T1, T2>>, is_copy_constructible, is_copy_constructible<Container, enable_if_t<all_of<
        std::is_copy_constructible<Container>,
        std::is_same<typename Container::value_type &, typename Container::reference>,
        // Avoid infinite recursion
        negation<std::is_same<Container, typename Container::value_type>>
    >::value>> (+5 more)

### Community 164 - "TEST_SUBMODULE"
Cohesion: 0.20
Nodes (7): call_policies, m, CustomGuard, enabled, DependentGuard, enabled, TEST_SUBMODULE()

### Community 165 - "Detail of Packages"
Cohesion: 0.17
Nodes (11): 1. Stand controller, 2. Position and pose publisher, Build, Dependencies, Detail of Packages, Introduction, Packages:, The description of robots: (+3 more)

### Community 166 - "Safety"
Cohesion: 0.17
Nodes (12): Safety, Calf_max, Calf_min, Hip_max, Hip_min, PositionLimit, PositionProtect, PowerProtect (+4 more)

### Community 167 - "T"
Cohesion: 0.15
Nodes (22): arg::operator=(), arg_v, descr, value, call(), call_impl(), cast_op(), cast_ref() (+14 more)

### Community 168 - "any_container"
Cohesion: 0.20
Nodes (10): any_container, v, constexpr_first(), constexpr_sum(), first(), Container, initializer_list, It (+2 more)

### Community 169 - "instance"
Cohesion: 0.11
Nodes (17): error_scope, trace, type, value, PyObject, instance, allocate_layout, deallocate_layout (+9 more)

### Community 170 - "unchecked_reference"
Cohesion: 0.18
Nodes (10): conditional_t, enable_if_t, size(), unchecked_reference, data_, dims_, Dynamic, shape_ (+2 more)

### Community 171 - "test_buffers.py"
Cohesion: 0.17
Nodes (4): skipif, SquareMatrix is derived from Matrix and inherits the buffer protocol, test_ctypes_from_buffer(), test_inherited_protocol()

### Community 172 - "Closed-Loop-Simulation für G1 + Dex3 in Isaac Lab — Implementierungsplan"
Cohesion: 0.17
Nodes (12): 0. Ausgangs-Spec (aus `examples/G1_DEX3/g1_dex3_config.py`), 10. Voraussetzung vor Phase A, 1. Architektur, 2. Laufumgebung auf KISSKI, 3. Roboter-Asset (kritischer Pfad), 4. Kameras (zweitkritischer Pfad), 5. Szene & Task, 6. Control-Loop (Kern des Sim-Clients) (+4 more)

### Community 173 - "Train-Test-Split (80/20)"
Cohesion: 0.17
Nodes (12): 1. `data/unitreerobotics/G1_Dex3_BlockStacking_Dataset/meta/info.json`, 2. `app/Groot-1.6/gr00t/data/dataset/lerobot_episode_loader.py`, 3. `app/Groot-1.6/gr00t/data/dataset/sharded_single_step_dataset.py`, 4. `app/Groot-1.6/gr00t/data/dataset/factory.py`, Aufteilung, Geänderte Dateien, Hinweis zur Reproduzierbarkeit, Motivation (+4 more)

### Community 174 - "Was für ein Training wird hier durchgeführt?"
Cohesion: 0.17
Nodes (12): 1. Lernparadigma: Imitation Learning (Behavior Cloning), 2. Basismodell: GR00T N1.6 (3B) — ein Vision-Language-Action-Modell, 3. Das Embodiment: Unitree G1 + Dex3 (28 DOF), 4. Ein- und Ausgaben, 5. Trainingsobjektiv: Flow Matching, 6. Datensatz, 7. Trainings-Hyperparameter — Lauf 1 (1× A100, historische Referenz), 8. Abgrenzung — was es *nicht* ist (+4 more)

### Community 175 - "Troubleshooting"
Cohesion: 0.17
Nodes (12): `createDLSSContext error` / Rendering schlägt fehl, Erfolgsrate bleibt 0, Reward explodiert, Historisch: DLSS-Upscaling (nicht die Ursache), Isaac-Sim-6.0-Migration (2026-08-07) — implementiert und seit 2026-08-08 auf Hardware bestätigt, `isaaclab nicht importierbar`, Kamerabilder gleichmäßig weiß — GELÖST (`runs/20260808/13` + `14`), Kopfkameras kalibrieren — Iteration 13 (2026-08-08), Renderprüfung offen, Segfault in `librtx.scenedb.plugin.so` / `carbOnPluginStartup` beim Start (RTX PRO 6000 Blackwell) (+4 more)

### Community 176 - "._force_camera_prim_orientations"
Cohesion: 0.18
Nodes (9): _matrix_to_quat(), _parse_rgb(), ndarray, _quat_to_matrix(), Gibt das Observation-Dict im GR00T-Eingabeformat zurück (numpy, CPU). Wird…, 0.75,0.73,0.70" -> (0.75, 0.73, 0.70). Leer/unparsbar -> None (Default…, (w, x, y, z) -> 3x3-Rotationsmatrix in Spaltenvektor-Konvention (v_welt = R @…, 3x3-Rotationsmatrix (Spaltenvektor-Konvention) -> (w, x, y, z). Shepperd-… (+1 more)

### Community 177 - "run_g1_gripper_sim_eval.py"
Cohesion: 0.27
Nodes (10): build_obs(), load_dims(), GR00T PolicyClient + Observation-Builder für die **UNITREE_G1**-Baseline (stock…, Baut das UNITREE_G1-Observation-Dict im Gr00tSimPolicyWrapper-Flat-Format.…, Lädt die Pro-Gruppe-Dimensionen (state/action) aus der Dump-JSON. Erwartetes…, main(), ndarray, Closed-Loop-Baseline-Eval für **un-finetuntes GR00T-N1.6-3B** auf dem **stock… (+2 more)

### Community 178 - "PolicyClient"
Cohesion: 0.23
Nodes (4): PolicyClient, ndarray, Sendet eine Observation und gibt einen (T, 16)-Action-Chunk zurück. Nur…, ZMQ-REQ-Client für den GR00T-Server im UNITREE_G1-Modus. get_action liefert…

### Community 179 - "optimize_groot_inference.py"
Cohesion: 0.11
Nodes (34): checkpoint_fingerprint(), gpu_fingerprint(), install_backend(), load_metadata(), metadata_path_for_engine(), Any, Path, Schneller, stabiler Fingerprint ohne mehrgigabyte-grosse Gewichte ganz zu lesen. (+26 more)

### Community 180 - "eigen.h"
Cohesion: 0.18
Nodes (9): eigen_encapsulate(), eigen_extract_stride, eigen_extract_stride<Eigen::Map<PlainObjectType, MapOptions, StrideType>>, eigen_extract_stride<Eigen::Ref<PlainObjectType, Options, StrideType>>, eigen_map_caster, name, eigen_ref_array(), load() (+1 more)

### Community 181 - "setup_helpers.py"
Cohesion: 0.25
Nodes (9): auto_cpp_level(), build_ext, has_flag(), Prepare and enter a temporary directory, cleanup when done, Return the flag if a flag name is supported on the specified compiler,…, Return the max supported C++ std level (17, 14, or 11). Returns latest on…, Customized build_ext that allows an auto-search for the highest supported C++…, Build extensions, injecting C++ std for Pybind11Extension if needed. (+1 more)

### Community 182 - "test_callbacks.py"
Cohesion: 0.20
Nodes (5): Test if passing a function pointer from C++ -> Python -> C++ yields the…, test_async_async_callbacks(), test_async_callbacks(), test_cpp_function_roundtrip(), test_function_signatures()

### Community 183 - "test_exceptions.py"
Cohesion: 0.17
Nodes (6): Tests nested (e.g. C++ -> Python -> C++) exception handling, test_custom(), test_error_already_set(), test_nested_throws(), test_python_alreadyset_in_destructor(), test_std_exception()

### Community 184 - "pybind11_fail"
Cohesion: 0.08
Nodes (19): PYBIND11_NOINLINE, pybind11_fail(), embedded_module, finalize_interpreter(), initialize_interpreter(), scoped_interpreter, is_valid, exception (+11 more)

### Community 185 - "Zweiten Docker-Container für die Sim bauen — Isaac Lab + GR00T-Client"
Cohesion: 0.18
Nodes (11): 0. Warum überhaupt ein zweiter Container?, 1. Strategie-Entscheidung: nicht „from scratch“, sondern auf NVIDIAs Image aufsetzen, 2. Voraussetzungen auf dem Laptop (einmalig), 3. Was kommt in den Container?, 4. Dockerfile, 5. Bauen und pushen (lokal, Laptop), 6. SIF auf KISSKI ziehen (einmalig, Login-Node), 7. Beide Container in einem SLURM-Job starten (+3 more)

### Community 186 - "Multi-GPU-Training (bis zu 4× A100)"
Cohesion: 0.18
Nodes (11): 1. Launcher auf `torchrun` umgestellt — [`Training/scripts/run_finetuning.sh`](../../Training/scripts/run_finetuning.sh), 2. SLURM-Ressourcen + Defaults — [`Training/kisski_submit.sh`](../../Training/kisski_submit.sh), 3. (Optional) DDP statt DeepSpeed erzwingen, Multi-GPU-Training (bis zu 4× A100), Realistische Erwartung, Step-Äquivalenz: 44 000 (Batch 32) ↔ 175 000 (Batch 8), Umgesetzte Änderungen, Verifikation (hier steckt die eigentliche Zeit, nicht im Coden) (+3 more)

### Community 187 - "3. Priorisierte Schritte"
Cohesion: 0.18
Nodes (11): 1. Wo das Projekt steht, 2. Ein Widerspruch in den Docs, vor dem nächsten Lauf zu klären, 3. Priorisierte Schritte, 4. Nebenbei offen (Komfort, nicht Fortschritt), 5. Was ausdrücklich *kein* nächster Schritt ist, Nächste Schritte, Schritt 1 — Checkpoint-Auswahl falsifizierbar machen ✅ *umgesetzt 2026-08-13*, Schritt 2 — Lauf 3 starten ✅ *umgesetzt 2026-08-13/14* (+3 more)

### Community 188 - "entrypoint_replay.sh"
Cohesion: 0.29
Nodes (8): err(), livestream_app_flags(), livestream_banner(), log(), ok(), PYTHONUNBUFFERED, entrypoint_replay.sh script, warn()

### Community 189 - "main"
Cohesion: 0.31
Nodes (10): find_checkpoints(), main(), parse_args(), Namespace, Path, Alle checkpoint-<step>-Verzeichnisse unterhalb von run_dir, nach Step sortiert.…, Positionen IM SPLIT — nicht zu verwechseln mit absoluten Episoden-Indizes., split.json, das run_finetuning*.sh beim Aktivieren des Splits ablegt. (+2 more)

### Community 190 - "GR00T N1.6 — Unitree G1 + DEX3 Dexterous Hand"
Cohesion: 0.20
Nodes (10): 1. Download a dataset, 2-camera datasets (use `modality_2cam.json`), 2. Fine-tune (4-camera example), 3. Run inference server, 4-camera datasets (use `modality_4cam.json`), Available Datasets, Design decisions, DEX3 Hand — Joint Layout (+2 more)

### Community 191 - "Type"
Cohesion: 0.33
Nodes (8): CType, cast(), cast_impl(), return_value_policy, Type, type_caster<Type, enable_if_t<is_eigen_dense_plain<Type>::value>>, name, value

### Community 192 - "UnitreeJointController::update"
Cohesion: 0.22
Nodes (9): effortLimits, getGains, positionLimits, velocityLimits, Duration, UnitreeJointController::update(), computeTorque(), computeVel() (+1 more)

### Community 193 - "unitree_ros_to_real/README.md"
Cohesion: 0.20
Nodes (9): Build, Configuration, Dependencies, Environment, Introduction, Notice, Packages:, Run the package (+1 more)

### Community 194 - "test_call_policies.py"
Cohesion: 0.22
Nodes (3): xfail, test_alive_gc(), test_keep_alive_argument()

### Community 195 - "type_caster<void>"
Cohesion: 0.13
Nodes (15): type, unique_ptr, implicit_cast(), move_only_holder_caster, name, type_caster, type_caster<std::unique_ptr<type, deleter>>, type_caster<void> (+7 more)

### Community 197 - "Basismodell-Referenz-Eval — Ergebnis (Pipeline-Validierung)"
Cohesion: 0.20
Nodes (10): 1. Setup, 2.1 Einzeltask (Top: `PlateToPlate`), 2.2 Aggregat über 24 Tasks (`full`-Lauf, 12/24 abgeschlossen) ★ entscheidend, 2.3 Abbruch (12/24) — manuell gestoppt, 2. Ergebnis, 3. Geklärte Nebenbefunde, 4. Was validiert ist — und was nicht, 5. Reproduktion (+2 more)

### Community 198 - "W&B-Auswertung — Run `g1_dex3_blockstacking_v1`"
Cohesion: 0.20
Nodes (10): 1. Run-Status (Stand 2026-06-03, ~08:39 UTC), 2. Trainingsdynamik — gesund ✅, 3. Over-/Underfitting — derzeit nicht messbar, 4. LR-Schedule — korrekt für 175k Steps ✅, 5.1 ⚠️ Batch-Size erhöht GPU-Auslastung (größter Hebel), 5.2 ⚠️ Eval ist konfiguriert, aber AUS — über ~21 h kein Generalisierungssignal, 5. Handlungsempfehlungen (für den **nächsten** Lauf, nicht den laufenden), 6. Sonstiges (unkritisch) (+2 more)

### Community 199 - "setup_and_train_Container-build.sh"
Cohesion: 0.40
Nodes (9): check_cmd(), COMPOSE_FILE, die(), err(), log(), ok(), run(), setup_and_train_Container-build.sh script (+1 more)

### Community 200 - "internals.h"
Cohesion: 0.28
Nodes (6): type_index, translate_exception(), translate_local_exception(), type_equal_to, type_hash, exception_ptr

### Community 201 - "EigenConformable"
Cohesion: 0.25
Nodes (8): EigenConformable, cols, conformable, negativestrides, rows, stride, EigenDStride, EigenIndex

### Community 202 - "Portabilität — das Repo auf einem fremden Rechner betreiben"
Cohesion: 0.13
Nodes (15): 1. Was sich geändert hat, 2.1 IKR-Server (Docker, `server_rl_run.sh`), 2.2 KISSKI (Apptainer + SLURM), 2.3 Checkliste, 2. Migration — das alte Verhalten auf den eigenen Maschinen wiederherstellen, 3.1 Eigener GPU-Server (Docker) — Sim, Greif-Diagnose, RL, 3.2 Anderes KISSKI-Projekt, 3.3 Anderer HPC-Cluster (kein KISSKI) (+7 more)

### Community 203 - "__main__.py"
Cohesion: 0.42
Nodes (4): get_cmake_dir(), get_include(), main(), print_includes()

### Community 204 - "Pybind11Extension"
Cohesion: 0.28
Nodes (5): Pybind11Extension, The CXX standard level. If set, will add the required flags. If left at 0, it…, Build a C++11+ Extension module with pybind11. This automatically adds the…, _Extension, setter

### Community 206 - "dict"
Cohesion: 0.22
Nodes (4): dict, PythonCallInDestructor, d, dict_iterator

### Community 207 - "PartialStruct"
Cohesion: 0.15
Nodes (13): m, PartialNestedStruct, a, dummy1, dummy2, PartialStruct, bool_, dummy2 (+5 more)

### Community 209 - "v3.8.6"
Cohesion: 0.22
Nodes (8): arm, Build, Cpp, Dependencies, Notice, Python, Run, v3.8.6

### Community 210 - "Dokumentation — Übersicht"
Cohesion: 0.22
Nodes (9): Dokumentation — Übersicht, Ergebnisse & Evaluation, Externe Ressourcen, Projektstruktur, Querschnitt (Training + Simulation), Simulation (Closed-Loop-Eval in Isaac Lab), Submodul-Dokumentation (`app/Groot-1.6/examples/G1_DEX3/`), Training (Fine-tuning von GR00T N1.6) (+1 more)

### Community 211 - "blend"
Cohesion: 0.36
Nodes (8): Image, blend(), main(), make_panel(), Multipliziert die RGB-Kanäle mit den angegebenen Faktoren (0–1 dämpft, >1…, Alpha-Blend: real (grün-getönt) über sim (rot-getönt). Perfekte Überlappung →…, 3-Panel: Real | Overlay (50 %) | Sim — mit Beschriftung., tint()

### Community 212 - "g1_gripper_blockstack_env.py"
Cohesion: 0.25
Nodes (7): G1GripperBlockstackEnvCfg, G1GripperBlockstackSceneCfg, configclass, DirectRLEnvCfg, InteractiveSceneCfg, Block-Stacking-Umgebung für den **stock Unitree G1 + Dex1-Parallelgreifer**…, Szene: Roboter am Tisch mit 3 Würfeln (gespiegelt vom DEX3-Env).

### Community 213 - "TEST_SUBMODULE"
Cohesion: 0.25
Nodes (7): callbacks, UnregisteredType, m, TEST_SUBMODULE(), m, TEST_SUBMODULE(), pytypes

### Community 214 - ".load"
Cohesion: 0.25
Nodes (5): data(), row_major, S, make_stride(), Scalar

### Community 215 - "ParallelCompile"
Cohesion: 0.32
Nodes (3): ParallelCompile, Make a parallel compile function. Inspired by…, Builds a function object usable as distutils.ccompiler.CCompiler.compile.

### Community 216 - "Widget"
Cohesion: 0.32
Nodes (5): string, PyWidget, Widget, message, the_answer

### Community 217 - "test_modules.py"
Cohesion: 0.25
Nodes (4): Pydoc needs to be able to provide help() for everything inside a pybind11 module, Registering two things with the same name, test_duplicate_registration(), test_pydoc()

### Community 218 - "SimpleStruct"
Cohesion: 0.15
Nodes (13): NestedStruct, a, b, PackedStruct, bool_, float_, ldbl_, uint_ (+5 more)

### Community 219 - "Bewertung der Sim-Umsetzung — Sinnvoll & korrekt? (2026-06-05)"
Cohesion: 0.25
Nodes (8): 1. Die zentrale Erkenntnis, 2. Konsequenz für die aktuelle Strategie (wichtigster Punkt), 3. Die richtige Bewertungsmethode (bereits vorhanden), 4. Was die Recherche *entlastet* hat, 5. Konkrete Code-Befunde (Handlungsbedarf), 6. Empfehlung, Bewertung der Sim-Umsetzung — Sinnvoll & korrekt? (2026-06-05), Quellenübersicht

### Community 220 - "GPU-Eignung für die Isaac-Sim-Closed-Loop-Sim auf GWDG"
Cohesion: 0.25
Nodes (8): 1. Das Problem: Isaac Sim braucht RT-Cores, 2. GPU-Landschaft auf GWDG, 3. Die Lösung: `jupyter.hpc.gwdg.de` (RTX 5000), 4. Konsequenz für die Zwei-Container-Architektur, 5. Empfohlenes Vorgehen, Caveats, GPU-Eignung für die Isaac-Sim-Closed-Loop-Sim auf GWDG, Quellen

### Community 221 - "Optionale Trainings-Features (getrennt schaltbar)"
Cohesion: 0.25
Nodes (8): Checkpoint-Auswahl nach dem Lauf ([`checkpoint_sweep.py`](../../Training/scripts/checkpoint_sweep.py)), Co-Training: gerenderten Datensatz dazumischen (`USE_COTRAIN=1`), Konfiguration über Env-Vars, Live-Ansicht des Laufs (nur im Sim-Image, opt-in), LIVE-Variante: Isaac-Sim-Viewport statt Videos (nur im Sim-Image, opt-in), Namespace des Laufs — ⚠️ der Fork setzt ungefragt fort, Optionale Trainings-Features (getrennt schaltbar), RL-Env-Vars (nur im Sim-Image, `entrypoint_rl.sh`)

### Community 222 - "Fixes aus dem ersten Trainingsdurchlauf"
Cohesion: 0.25
Nodes (8): 1. Ausgangslage (kurz), 2. Umgesetzte Fixes (Sim-Asset/Env, kein Retraining), 3. Neue Werkzeuge, 4. `BLACK_HANDS`-Mechanik (selbstheilend), 5. Anwenden & verifizieren (Sim-Container, RT-Core-GPU), 6. Offene Punkte / noch zu verifizieren, 7. Aus der Auswertung noch NICHT adressiert, Fixes aus dem ersten Trainingsdurchlauf

### Community 223 - "extract_block_layout.py"
Cohesion: 0.09
Nodes (35): G1Dex3CameraCfg, look_at_world_quat(), PinholeCamera, ndarray, quat_to_matrix(), Projektion und Rückprojektion für eine der weltfesten Kameras. Bildachsen in…, Eine der weltfesten Kameras aus der Konfiguration bauen. Nur die drei…, Weltpunkte (…,3) → Pixel (…,2) als (u, v). Punkte hinter der Kamera: NaN. (+27 more)

### Community 224 - "g1_gripper_cfg.py"
Cohesion: 0.29
Nodes (6): G1GripperCameraCfg, look_at_world_quat(), configclass, Articulation-Konfiguration für den **stock Unitree G1 mit…, Quaternion (w, x, y, z) für eine Kamera in Isaac-Lab-``convention="world"``.…, Kamera-Posen — ``UNITREE_G1`` erwartet GENAU eine Kamera ``ego_view``.

### Community 225 - "update_sim_image.sh"
Cohesion: 0.50
Nodes (7): err(), fatal(), log(), ok(), run(), update_sim_image.sh script, warn()

### Community 226 - "entrypoint.sh"
Cohesion: 0.50
Nodes (7): data_present(), err(), log(), ok(), entrypoint.sh script, trap_err(), warn()

### Community 227 - "setup_and_train_DockerHub-pull.sh"
Cohesion: 0.46
Nodes (7): err(), fatal(), invoke_cmd(), log(), ok(), setup_and_train_DockerHub-pull.sh script, warn()

### Community 228 - "update_image.sh"
Cohesion: 0.50
Nodes (7): err(), exit_fatal(), invoke_cmd(), log(), ok(), update_image.sh script, warn()

### Community 229 - ".init"
Cohesion: 0.43
Nodes (5): arg, arg_v, process_attribute<arg>, process_attribute<arg_v>, process_kw_only_arg()

### Community 230 - "_Handler"
Cohesion: 0.38
Nodes (3): BaseHTTPRequestHandler, _Handler, HTTP-Handler. `view` wird per Subklasse beim Serverstart gebunden.

### Community 231 - "Usages"
Cohesion: 0.29
Nodes (6): Gazebo, [MuJoCo](https://github.com/google-deepmind/mujoco)(recommend), Overview, RViz, Unitree H1 Description (URDF & MJCF), Usages

### Community 232 - "body.cpp"
Cohesion: 0.52
Nodes (6): motion_init(), moveAllPosition(), paramInit(), sendServoCmd(), stand(), main()

### Community 233 - "functional.h"
Cohesion: 0.25
Nodes (5): _thread, cast(), Func, return_value_policy, load()

### Community 234 - "argument_record"
Cohesion: 0.29
Nodes (6): argument_record, convert, descr, name, none, value

### Community 235 - ".load"
Cohesion: 0.36
Nodes (6): type_list, forward_like(), U, load_alternative(), variant_caster<V<Ts...>>, forwarded_type

### Community 236 - "DerivedWidget"
Cohesion: 0.29
Nodes (4): m, DerivedWidget, PYBIND11_EMBEDDED_MODULE(), widget_module

### Community 237 - "DtypeSizeCheck"
Cohesion: 0.29
Nodes (7): string, DtypeSizeCheck, name, size_cpp, size_numpy, get_dtype_size_check(), get_platform_dtype_size_checks()

### Community 238 - "Repository Guidelines"
Cohesion: 0.25
Nodes (7): Build, Test, and Development Commands, Coding Style & Naming Conventions, Commit & Pull Request Guidelines, Project Structure & Module Organization, Repository Guidelines, Security & Configuration, Testing Guidelines

### Community 239 - "Baseline-Closed-Loop-Test — stock Unitree G1 + Dex1-Greifer (UNITREE_G1)"
Cohesion: 0.33
Nodes (6): 1. Warum das eine OOD-Baseline ist (Erwartungsmanagement), 2. Architektur & neue Dateien, 3. ✅ TODO — vor dem ERSTEN Run zwingend erledigen, 4. Run ausführen (vast.ai), 5. Ergebnisse & Vergleich, Baseline-Closed-Loop-Test — stock Unitree G1 + Dex1-Greifer (UNITREE_G1)

### Community 240 - "W&B Offline-Sync auf KISSKI"
Cohesion: 0.29
Nodes (7): Bereits gesyncte Runs überspringen, Job starten (mit W&B), Live-Tracking während des Runs (inkrementeller Sync), Mehrere Runs auf einmal syncen, Nach dem Training: Sync zum W&B-Dashboard, W&B Offline-Sync auf KISSKI, Wo werden die Runs gespeichert?

### Community 241 - "entrypoint_rl.sh"
Cohesion: 0.57
Nodes (6): err(), log(), ok(), entrypoint_rl.sh script, trap_err(), warn()

### Community 242 - "run_trajectory"
Cohesion: 0.38
Nodes (6): main(), ndarray, Größte Spannweite (max-min über die Episode) je Gruppe, über deren Gelenke., Gibt (pred, gt, group_slices) für eine Trajektorie zurück., run_trajectory(), span_per_group()

### Community 243 - "phase_b_test.py"
Cohesion: 0.47
Nodes (5): ArticulationCfg, build_robot_cfg(), main(), Phase-B-Test: G1+Dex3 USD in Isaac Lab laden und Arm-Joints prüfen. Fertig…, run_test()

### Community 244 - "list"
Cohesion: 0.29
Nodes (4): process(), list, _values, list_iterator

### Community 245 - "typeid.h"
Cohesion: 0.60
Nodes (5): clean_type_id(), erase_all(), PYBIND11_NOINLINE, string, type_id()

### Community 246 - "array_info<std::array<T, N>>"
Cohesion: 0.22
Nodes (7): array_info<std::array<T, N>>, extent, extents, is_array, is_empty, npy_format_descriptor<T, enable_if_t<array_info<T>::is_array>>, name

### Community 247 - "test_async.py"
Cohesion: 0.47
Nodes (5): event_loop(), get_await_result(), fixture, test_await(), test_await_missing()

### Community 248 - "DtypeCheck"
Cohesion: 0.33
Nodes (6): DtypeCheck, numpy, pybind11, get_concrete_dtype_checks(), get_dtype_check(), dtype

### Community 249 - "parametrize"
Cohesion: 0.33
Nodes (6): parametrize, test_argument_conversions(), test_at_fail(), test_data(), test_index_offset(), filterwarnings

### Community 250 - "Läufe 33/34 (`runs/20260814/03`, `runs/20260814/04`): der TUNE_VISUAL-Checkpoint im Closed Loop"
Cohesion: 0.33
Nodes (6): Der Vergleich mit Lauf 31 (gleiches Zeitfenster, alter Checkpoint), Konsequenz, Lauf 34, Episode für Episode, Läufe 33/34 (`runs/20260814/03`, `runs/20260814/04`): der TUNE_VISUAL-Checkpoint im Closed Loop, Was sich nicht bewegt hat, Zuordnung: es war der Checkpoint

### Community 251 - "Schritt 2 — Domain-Gap neu messen (`server_rl_run.sh gap`)"
Cohesion: 0.33
Nodes (6): Ergebnis (`runs/20260808/19`) — der Gap sitzt in der Belichtung, nicht in der Geometrie, Ergebnis (`runs/20260808/21`) — halb gemessen, und die Hälfte war schon gelöst, Ergebnis (`runs/20260808/22`) — Albedo bestätigt, Kriterium um 0,0056 verfehlt, Schritt 2 — Domain-Gap neu messen (`server_rl_run.sh gap`), Sweep-Ergebnis (`runs/20260808/20`) — Belichtung ist NICHT der Hebel, Was es stattdessen ist: Albedo und Hintergrund

### Community 252 - "LaTeX — Projektarbeit (Ausarbeitung)"
Cohesion: 0.33
Nodes (5): Kompilieren, LaTeX — Projektarbeit (Ausarbeitung), Nächste Schritte (TODO), Struktur, Zweck

### Community 253 - "Projektarbeit Humanoider Roboter"
Cohesion: 0.33
Nodes (6): Dokumentation, GR00T N1.6 Fine-tuning — Unitree G1 mit DEX3-Hand, Projektarbeit Humanoider Roboter, Schnellstart, Voraussetzungen (Kurzfassung), Weiterführende Ressourcen

### Community 254 - "void_caster"
Cohesion: 0.33
Nodes (6): nullptr_t, type_caster<std::nullptr_t>, void_caster, type_caster<std::experimental::nullopt_t>, type_caster<std::nullopt_t>, nullopt_t

### Community 255 - "check_action_norm.py"
Cohesion: 0.53
Nodes (5): close(), first_row(), fmt(), main(), Erste Zeile einer evtl. verschachtelten Liste — relative Stats sind (T, D).

### Community 256 - "b2_description/README.md"
Cohesion: 0.40
Nodes (4): b2_urdf, Build the library, Run the library, support issac

### Community 257 - "b2w_description/README.md"
Cohesion: 0.40
Nodes (4): b2w_urdf, Build the library, Run the library, support issac

### Community 258 - "go2_description/README.md"
Cohesion: 0.40
Nodes (4): Build the library, go2_urdf, Run the library, When used for isaac gym or other similiar engine

### Community 259 - "doc"
Cohesion: 0.50
Nodes (3): doc, value, process_attribute<doc>

### Community 260 - "is_method"
Cohesion: 0.50
Nodes (3): is_method, class_, process_attribute<is_method>

### Community 261 - "module_local"
Cohesion: 0.50
Nodes (3): module_local, value, process_attribute<module_local>

### Community 262 - "name"
Cohesion: 0.50
Nodes (3): name, value, process_attribute<name>

### Community 263 - "scope"
Cohesion: 0.50
Nodes (3): process_attribute<scope>, scope, value

### Community 264 - "sibling"
Cohesion: 0.50
Nodes (3): process_attribute<sibling>, sibling, value

### Community 265 - "format_descriptor<T, detail::enable_if_t<std::is_arithmetic<T>::value>>"
Cohesion: 0.40
Nodes (4): format_descriptor<T, detail::enable_if_t<std::is_arithmetic<T>::value>>, c, value, string

### Community 266 - "size_in_ptrs"
Cohesion: 0.40
Nodes (4): PyObject_HEAD, instance_simple_holder_in_ptrs(), log2(), size_in_ptrs()

### Community 267 - "operator()"
Cohesion: 0.40
Nodes (5): Return, true_type, is_input_iterator<T, void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>, is_instantiation<Class, Class<Us...>>, operator()()

### Community 268 - "setup.py"
Cohesion: 0.33
Nodes (4): get_and_replace(), Prepare a temporary directory, cleanup when done, SDist, TemporaryDirectory()

### Community 269 - "MyBase"
Cohesion: 0.50
Nodes (3): unique_ptr, MyBase, MyDerived

### Community 270 - "PYBIND11_EMBEDDED_MODULE"
Cohesion: 0.40
Nodes (3): m, test_cmake_build, PYBIND11_EMBEDDED_MODULE()

### Community 272 - "AliasedHasOpNewDelSize"
Cohesion: 0.33
Nodes (5): AliasedHasOpNewDelSize, i, PyAliasedHasOpNewDelSize, j, uint64_t

### Community 273 - "EnumStruct"
Cohesion: 0.40
Nodes (5): EnumStruct, e1, e2, E1, E2

### Community 274 - "SimpleStructReordered"
Cohesion: 0.40
Nodes (5): SimpleStructReordered, bool_, float_, ldbl_, uint_

### Community 275 - "test_numpy_vectorize.cpp"
Cohesion: 0.50
Nodes (4): m, my_func(), TEST_SUBMODULE(), numpy_vectorize

### Community 277 - "split_own_args"
Cohesion: 0.40
Nodes (3): Namespace, Die beiden eigenen Flags vor tyro abfangen. tyro baut sein CLI aus den Feldern…, split_own_args()

### Community 278 - "Unitree G1 Description (URDF & MJCF)"
Cohesion: 0.50
Nodes (3): Overview, Unitree G1 Description (URDF & MJCF), Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

### Community 279 - "Unitree H1_2 Description (URDF & MJCF)"
Cohesion: 0.50
Nodes (3): Overview, Unitree H1_2 Description (URDF & MJCF), Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

### Community 281 - "setup"
Cohesion: 0.67
Nodes (3): generate_doxygen_xml(), Add hook for building doxygen xml when needed, setup()

### Community 282 - "select_indices_impl<index_sequence<IPrev...>, I, B, Bs...>"
Cohesion: 0.40
Nodes (5): conditional_t, select_indices_impl, select_indices_impl<index_sequence<IPrev...>, I, B, Bs...>, B, I

### Community 283 - "get_shared_data"
Cohesion: 0.67
Nodes (4): get_shared_data(), PYBIND11_NOINLINE, string, set_shared_data()

### Community 284 - "pair"
Cohesion: 0.50
Nodes (3): pair, PyObject, override_hash

### Community 285 - "TEST_SUBMODULE"
Cohesion: 0.50
Nodes (3): m, TEST_SUBMODULE(), docstring_options

### Community 286 - "TEST_SUBMODULE"
Cohesion: 0.50
Nodes (3): m, TEST_SUBMODULE(), union_

### Community 287 - "GR00T-N1.6: ONNX/TensorRT und schnelleres Sim-Rendering"
Cohesion: 0.29
Nodes (6): Backend auswählen, Einmaliges Setup auf dem IKR-Server, Engine bauen und automatisch testen, GR00T-N1.6: ONNX/TensorRT und schnelleres Sim-Rendering, Optimierte Eval mit Webstream, Wichtige Grenzen

### Community 288 - "Training — Fine-tuning von GR00T N1.6"
Cohesion: 0.50
Nodes (4): Die vier Wege im Überblick, Einstieg, Training — Fine-tuning von GR00T N1.6, Verwandte Dokumentation

### Community 289 - "RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung"
Cohesion: 0.50
Nodes (4): Pfad B — eigener Docker-GPU-Server mit RT-Cores ★ empfohlen, wenn verfügbar, RL-Fine-tuning (FPO) — Schritt-für-Schritt-Anleitung, Schnellstart, Überblick: Was läuft wo

### Community 290 - "kisski_replay_submit.sh"
Cohesion: 0.40
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, kisski_replay_submit.sh script

### Community 291 - "kisski_robocasa_ref_submit.sh"
Cohesion: 0.40
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, kisski_robocasa_ref_submit.sh script

### Community 292 - "kisski_sim_submit.sh"
Cohesion: 0.40
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, kisski_sim_submit.sh script

### Community 293 - "_enable_hf_transfer"
Cohesion: 0.67
Nodes (3): _enable_hf_transfer(), main(), hf_transfer aktivieren, falls verfügbar (Rust-Backend, deutlich schneller).

### Community 294 - "kisski_open_loop_eval.sh"
Cohesion: 0.40
Nodes (3): APPTAINER_CACHEDIR, APPTAINER_TMPDIR, kisski_open_loop_eval.sh script

### Community 295 - "download_data.sh"
Cohesion: 0.50
Nodes (3): HF_TOKEN, HUGGING_FACE_HUB_TOKEN, download_data.sh script

### Community 297 - "CustomOperatorNew"
Cohesion: 0.33
Nodes (5): CustomOperatorNew, a, b, EIGEN_MAKE_ALIGNED_OPERATOR_NEW, Matrix4d

### Community 301 - "A_Tpl"
Cohesion: 0.40
Nodes (3): A_METHODS, A_Repeat, A_Tpl

### Community 302 - "arg"
Cohesion: 0.50
Nodes (4): arg, flag_noconvert, flag_none, operator"" _a()

### Community 304 - "summarize_run"
Cohesion: 0.70
Nodes (4): main(), mean(), Path, summarize_run()

### Community 305 - "builtin_exception"
Cohesion: 0.67
Nodes (3): builtin_exception, set_error, runtime_error

### Community 306 - "exactly_one"
Cohesion: 0.67
Nodes (3): exactly_one, found, index

### Community 308 - "is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>>"
Cohesion: 0.67
Nodes (3): is_fmt_numeric<T, enable_if_t<std::is_arithmetic<T>::value>>, index, value

### Community 310 - "xfail"
Cohesion: 0.67
Nodes (3): xfail, test_array_create_and_resize(), test_dtype_refcount_leak()

### Community 313 - "test_modules.cpp"
Cohesion: 0.50
Nodes (3): m, TEST_SUBMODULE(), modules

### Community 316 - "ComplexStruct"
Cohesion: 0.50
Nodes (4): ComplexStruct, cdbl, cflt, complex

### Community 317 - "policy_latency.py"
Cohesion: 0.67
Nodes (3): build_synthetic_obs(), main(), Observation im flat-key Format des Gr00tSimPolicyWrapper (wie…

## Knowledge Gaps
- **1325 isolated node(s):** `compare_domain_gap.sh script`, `kisski_replay_submit.sh script`, `APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`, `kisski_robocasa_ref_submit.sh script` (+1320 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `object` connect `object` to `handle`, `cast.h`, `cpp_function`, `pybind11.h`, `function_call`, `iterator`, `array`, `sequence`, `T`, `return_value_policy`, `OstreamRedirect`, `dtype`, `pybind11_fail`, `str`, `conftest.py`, `dict`, `ParallelCompile`, `Capture`, `index_sequence`, `pytypes.h`, `list`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `PYBIND11_MODULE()` connect `pybind11_tests.cpp` to `pybind11_tests`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `UnitreeJointController` connect `UnitreeJointController` to `UnitreeJointController::update`, `convert.h`, `joint_controller.h`, `name_space`, `vector`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `object` (e.g. with `.dec_ref()` and `.inc_ref()`) actually correct?**
  _`object` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 56 inferred relationships involving `pybind11_fail()` (e.g. with `.init()` and `process_kw_only_arg()`) actually correct?**
  _`pybind11_fail()` has 56 INFERRED edges - model-reasoned connections that need verification._
- **What connects `compare_domain_gap.sh script`, `kisski_replay_submit.sh script`, `APPTAINER_CACHEDIR` to the rest of the system?**
  _1325 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `handle` be split into smaller, more focused modules?**
  _Cohesion score 0.09343200740055504 - nodes in this community are weakly interconnected._