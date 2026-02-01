import json
from typing import Dict, Optional
from backend.agent import LLMClient

class QueryExpander:
    """
    Module to expand user queries using LLM for better literature search results
    """
    
    def __init__(self):
        """
        Initialize QueryExpander
        """
        pass
    
    def expand_query(
        self,
        user_query: str,
        llm_provider: str = "openai",
        api_key: str = ""
    ) -> Dict:
        """
        Expand user query using LLM
        
        Args:
            user_query: Original user query
            llm_provider: LLM provider to use
            api_key: API key for the LLM provider
            
        Returns:
            Dict with expanded search query and extracted filters
        """
        system_prompt = """
        You are an expert research assistant specializing in academic paper search.
        Your task is to transform user queries into optimized search strings for academic databases like OpenAlex and arXiv.
        
        Process:
        1. Expand abbreviations (e.g., VLA → Vision-Language-Action)
        2. Add synonyms and related terms
        3. Extract venue names (e.g., CVPR, ICML) and move them to filters
        4. Identify field-specific terminology
        5. Create a comprehensive search string
        
        Output format:
        {
            "search_query": "Expanded search string with synonyms and related terms",
            "venue_filter": "Extracted venue name or empty string",
            "year_range": "YYYY-YYYY or empty string",
            "domain": "Research domain category",
            "explanation": "Brief explanation of the expansion process"
        }
        
        Examples:
        
        Input: "CVPR VLA 自动驾驶"
        Output:
        {
            "search_query": "(Vision-Language-Action OR VLA OR Vision Language Action OR Multimodal LLM for Robotics) AND (Autonomous Driving OR Self-driving OR AD OR End-to-end driving OR Autonomous Vehicles)",
            "venue_filter": "CVPR",
            "year_range": "",
            "domain": "Computer Vision, Robotics",
            "explanation": "Expanded VLA to Vision-Language-Action and related terms, expanded 自动驾驶 to Autonomous Driving and synonyms, extracted CVPR as venue filter"
        }
        
        Input: "Mamba architecture for vision"
        Output:
        {
            "search_query": "(Mamba architecture OR Mamba) AND (computer vision OR vision OR visual recognition OR image processing)",
            "venue_filter": "",
            "year_range": "",
            "domain": "Computer Vision, Machine Learning",
            "explanation": "Added Mamba architecture variations and computer vision related terms"
        }
        
        Input: "2024 NIPS transformer models for protein folding"
        Output:
        {
            "search_query": "(transformer models OR transformers) AND (protein folding OR protein structure prediction OR protein modeling)",
            "venue_filter": "NIPS",
            "year_range": "2024",
            "domain": "Computational Biology, Machine Learning",
            "explanation": "Extracted NIPS as venue filter and 2024 as year, expanded transformer models and protein folding related terms"
        }
        """
        
        # If no API key, return basic expansion but still extract venues
        if not api_key:
            venue_filter = self.extract_venues(user_query)
            return {
                "search_query": user_query,
                "venue_filter": venue_filter,
                "year_range": "",
                "domain": "General",
                "explanation": "No API key provided, using original query but extracted venue"
            }
        
        try:
            # Create LLM client
            client = LLMClient(api_key=api_key, provider=llm_provider)
            
            # Generate response
            response = client.generate(system_prompt, user_query)
            
            if response:
                try:
                    # Parse JSON response
                    expanded_query = json.loads(response)
                    return expanded_query
                except json.JSONDecodeError:
                    # If JSON parsing fails, return original query
                    pass
        except Exception as e:
            print(f"Error expanding query: {e}")
        
        # Return original query if any error occurs
        # But still extract venues from the original query
        venue_filter = self.extract_venues(user_query)
        
        return {
            "search_query": user_query,
            "venue_filter": venue_filter,
            "year_range": "",
            "domain": "General",
            "explanation": "Query expansion failed, using original query but extracted venue"
        }
    
    def extract_venues(self, user_query: str) -> str:
        """
        Extract venue names from user query
        
        Args:
            user_query: User query
            
        Returns:
            Extracted venue name or empty string
        """
        # Common CS conference and journal names
        venues = [
            "CVPR", "ICCV", "ECCV", "ICML", "NeurIPS", "NIPS", "ICLR",
            "AAAI", "IJCAI", "ACL", "EMNLP", "NAACL", "ICRA", "IROS",
            "ICCVW", "CVPRW", "ICMLW", "NeurIPSW", "ICLRW",
            "TPAMI", "IJCV", "JMLR", "JAIR", "ACM TOG", "ACM SIGGRAPH"
        ]
        
        query_lower = user_query.lower()
        for venue in venues:
            if venue.lower() in query_lower:
                return venue
        
        return ""
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand common abbreviations in scientific queries
        
        Args:
            query: Original query
            
        Returns:
            Query with expanded abbreviations
        """
        abbreviations = {
            "VLA": "Vision-Language-Action",
            "AD": "Autonomous Driving",
            "LLM": "Large Language Model",
            "CV": "Computer Vision",
            "NLP": "Natural Language Processing",
            "RL": "Reinforcement Learning",
            "ML": "Machine Learning",
            "DL": "Deep Learning",
            "AI": "Artificial Intelligence",
            "IoT": "Internet of Things",
            "GPU": "Graphics Processing Unit",
            "CPU": "Central Processing Unit",
            "RAM": "Random Access Memory",
            "ROM": "Read-Only Memory",
            "API": "Application Programming Interface",
            "URL": "Uniform Resource Locator",
            "HTTP": "Hypertext Transfer Protocol",
            "HTTPS": "Hypertext Transfer Protocol Secure",
            "JSON": "JavaScript Object Notation",
            "XML": "Extensible Markup Language",
            "SQL": "Structured Query Language",
            "NoSQL": "Not Only SQL",
            "REST": "Representational State Transfer",
            "GraphQL": "Graph Query Language",
            "OAuth": "Open Authorization",
            "JWT": "JSON Web Token",
            "MLP": "Multilayer Perceptron",
            "CNN": "Convolutional Neural Network",
            "RNN": "Recurrent Neural Network",
            "LSTM": "Long Short-Term Memory",
            "GRU": "Gated Recurrent Unit",
            "Transformer": "Transformer",
            "BERT": "Bidirectional Encoder Representations from Transformers",
            "GPT": "Generative Pre-trained Transformer",
            "ViT": "Vision Transformer",
            "Swin": "Swin Transformer",
            "MAE": "Masked Autoencoder",
            "DINO": "Self-distillation with no labels",
            "MoCo": "Momentum Contrast",
            "SimCLR": "Simple Contrastive Learning of Visual Representations",
            "BYOL": "Bootstrap Your Own Latent",
            "CLIP": "Contrastive Language-Image Pre-training",
            "DALL-E": "DALL-E",
            "Stable Diffusion": "Stable Diffusion",
            "GAN": "Generative Adversarial Network",
            "VAE": "Variational Autoencoder",
            "AE": "Autoencoder",
            "GANs": "Generative Adversarial Networks",
            "VAEs": "Variational Autoencoders",
            "AEs": "Autoencoders",
            "RLHF": "Reinforcement Learning with Human Feedback",
            "RLOA": "Reinforcement Learning from Online Action",
            "RLE": "Reinforcement Learning from Expert Demonstrations",
            "MDP": "Markov Decision Process",
            "POMDP": "Partially Observable Markov Decision Process",
            "Q-learning": "Q-learning",
            "SARSA": "SARSA",
            "DQN": "Deep Q-Network",
            "DDPG": "Deep Deterministic Policy Gradient",
            "PPO": "Proximal Policy Optimization",
            "TRPO": "Trust Region Policy Optimization",
            "SAC": "Soft Actor-Critic",
            "TD3": "Twin Delayed Deep Deterministic Policy Gradients",
            "MPC": "Model Predictive Control",
            "LQR": "Linear Quadratic Regulator",
            "PID": "Proportional-Integral-Derivative",
            "ROS": "Robot Operating System",
            "URDF": "Unified Robot Description Format",
            "Gazebo": "Gazebo",
            "V-REP": "Virtual Robot Experimentation Platform",
            "CoppeliaSim": "CoppeliaSim",
            "Webots": "Webots",
            "MuJoCo": "Multi-Joint dynamics with Contact",
            "PyBullet": "PyBullet",
            "OpenAI Gym": "OpenAI Gym",
            "Gymnasium": "Gymnasium",
            "RLlib": "RLlib",
            "Stable Baselines": "Stable Baselines",
            "TensorFlow": "TensorFlow",
            "PyTorch": "PyTorch",
            "JAX": "JAX",
            "NumPy": "NumPy",
            "Pandas": "Pandas",
            "Matplotlib": "Matplotlib",
            "Seaborn": "Seaborn",
            "Scikit-learn": "Scikit-learn",
            "Keras": "Keras",
            "FastAI": "FastAI",
            "Hugging Face": "Hugging Face",
            "Transformers": "Transformers",
            "Diffusers": "Diffusers",
            "OpenCV": "OpenCV",
            "PIL": "Python Imaging Library",
            "Scipy": "Scipy",
            "NetworkX": "NetworkX",
            "DGL": "Deep Graph Library",
            "PyTorch Geometric": "PyTorch Geometric",
            "cuDNN": "CUDA Deep Neural Network library",
            "cuBLAS": "CUDA Basic Linear Algebra Subprograms",
            "TensorRT": "TensorRT",
            "ONNX": "Open Neural Network Exchange",
            "TorchScript": "TorchScript",
            "TensorFlow Lite": "TensorFlow Lite",
            "PyTorch Mobile": "PyTorch Mobile",
            "ONNX Runtime": "ONNX Runtime",
            "TVM": "Tensor Virtual Machine",
            "MLIR": "Multi-Level Intermediate Representation",
            "LLVM": "Low Level Virtual Machine",
            "GCC": "GNU Compiler Collection",
            "Clang": "Clang",
            "CMake": "CMake",
            "Make": "Make",
            "Docker": "Docker",
            "Kubernetes": "Kubernetes",
            "AWS": "Amazon Web Services",
            "Azure": "Microsoft Azure",
            "GCP": "Google Cloud Platform",
            "CPU": "Central Processing Unit",
            "GPU": "Graphics Processing Unit",
            "TPU": "Tensor Processing Unit",
            "NPU": "Neural Processing Unit",
            "VPU": "Vision Processing Unit",
            "FPGA": "Field-Programmable Gate Array",
            "ASIC": "Application-Specific Integrated Circuit",
            "RAM": "Random Access Memory",
            "ROM": "Read-Only Memory",
            "SSD": "Solid-State Drive",
            "HDD": "Hard Disk Drive",
            "PCIe": "Peripheral Component Interconnect Express",
            "USB": "Universal Serial Bus",
            "HDMI": "High-Definition Multimedia Interface",
            "DisplayPort": "DisplayPort",
            "Thunderbolt": "Thunderbolt",
            "Ethernet": "Ethernet",
            "Wi-Fi": "Wi-Fi",
            "Bluetooth": "Bluetooth",
            "5G": "5G",
            "4G": "4G",
            "3G": "3G",
            "LTE": "Long-Term Evolution",
            "GSM": "Global System for Mobile Communications",
            "CDMA": "Code Division Multiple Access",
            "GPS": "Global Positioning System",
            "GNSS": "Global Navigation Satellite System",
            "IMU": "Inertial Measurement Unit",
            "LiDAR": "Light Detection and Ranging",
            "RADAR": "Radio Detection and Ranging",
            "SONAR": "Sound Navigation and Ranging",
            "Camera": "Camera",
            "Microphone": "Microphone",
            "Speaker": "Speaker",
            "Touchscreen": "Touchscreen",
            "Keyboard": "Keyboard",
            "Mouse": "Mouse",
            "Monitor": "Monitor",
            "Printer": "Printer",
            "Scanner": "Scanner",
            "Projector": "Projector",
            "Headphones": "Headphones",
            "Speakers": "Speakers",
            "Microphones": "Microphones",
            "Cameras": "Cameras",
            "Sensors": "Sensors",
            "Actuators": "Actuators",
            "Motors": "Motors",
            "Servos": "Servos",
            "Stepper Motors": "Stepper Motors",
            "DC Motors": "DC Motors",
            "AC Motors": "AC Motors",
            "Brushless DC Motors": "Brushless DC Motors",
            "Brushed DC Motors": "Brushed DC Motors",
            "Stepper Motor Drivers": "Stepper Motor Drivers",
            "Motor Controllers": "Motor Controllers",
            "H-Bridges": "H-Bridges",
            "PWM Controllers": "PWM Controllers",
            "Encoders": "Encoders",
            "Rotary Encoders": "Rotary Encoders",
            "Linear Encoders": "Linear Encoders",
            "Optical Encoders": "Optical Encoders",
            "Magnetic Encoders": "Magnetic Encoders",
            "Hall Effect Sensors": "Hall Effect Sensors",
            "Infrared Sensors": "Infrared Sensors",
            "Ultrasonic Sensors": "Ultrasonic Sensors",
            "Proximity Sensors": "Proximity Sensors",
            "Force Sensors": "Force Sensors",
            "Torque Sensors": "Torque Sensors",
            "Pressure Sensors": "Pressure Sensors",
            "Temperature Sensors": "Temperature Sensors",
            "Humidity Sensors": "Humidity Sensors",
            "Light Sensors": "Light Sensors",
            "Gas Sensors": "Gas Sensors",
            "Chemical Sensors": "Chemical Sensors",
            "Biosensors": "Biosensors",
            "Medical Sensors": "Medical Sensors",
            "Industrial Sensors": "Industrial Sensors",
            "Automotive Sensors": "Automotive Sensors",
            "Aerospace Sensors": "Aerospace Sensors",
            "Defense Sensors": "Defense Sensors",
            "Consumer Electronics Sensors": "Consumer Electronics Sensors",
            "IoT Sensors": "IoT Sensors",
            "Smart City Sensors": "Smart City Sensors",
            "Smart Home Sensors": "Smart Home Sensors",
            "Agricultural Sensors": "Agricultural Sensors",
            "Environmental Sensors": "Environmental Sensors",
            "Weather Sensors": "Weather Sensors",
            "Climate Sensors": "Climate Sensors",
            "Air Quality Sensors": "Air Quality Sensors",
            "Water Quality Sensors": "Water Quality Sensors",
            "Soil Sensors": "Soil Sensors",
            "Plant Sensors": "Plant Sensors",
            "Animal Sensors": "Animal Sensors",
            "Human Sensors": "Human Sensors",
            "Wearable Sensors": "Wearable Sensors",
            "Implantable Sensors": "Implantable Sensors",
            "Ingestible Sensors": "Ingestible Sensors",
            "Non-invasive Sensors": "Non-invasive Sensors",
            "Invasive Sensors": "Invasive Sensors",
            "Contact Sensors": "Contact Sensors",
            "Non-contact Sensors": "Non-contact Sensors",
            "Wireless Sensors": "Wireless Sensors",
            "Wired Sensors": "Wired Sensors",
            "Passive Sensors": "Passive Sensors",
            "Active Sensors": "Active Sensors",
            "Digital Sensors": "Digital Sensors",
            "Analog Sensors": "Analog Sensors",
            "MEMS Sensors": "Micro-Electro-Mechanical Systems Sensors",
            "NEMS Sensors": "Nano-Electro-Mechanical Systems Sensors",
            "CMOS Sensors": "Complementary Metal-Oxide-Semiconductor Sensors",
            "CCD Sensors": "Charge-Coupled Device Sensors",
            "Image Sensors": "Image Sensors",
            "Vision Sensors": "Vision Sensors",
            "Camera Sensors": "Camera Sensors",
            "Video Sensors": "Video Sensors",
            "Audio Sensors": "Audio Sensors",
            "Microphone Sensors": "Microphone Sensors",
            "Sound Sensors": "Sound Sensors",
            "Vibration Sensors": "Vibration Sensors",
            "Accelerometers": "Accelerometers",
            "Gyroscopes": "Gyroscopes",
            "Magnetometers": "Magnetometers",
            "Inertial Navigation Systems": "Inertial Navigation Systems",
            "GPS Receivers": "GPS Receivers",
            "GNSS Receivers": "GNSS Receivers",
            "Navigation Sensors": "Navigation Sensors",
            "Position Sensors": "Position Sensors",
            "Location Sensors": "Location Sensors",
            "Tracking Sensors": "Tracking Sensors",
            "Motion Sensors": "Motion Sensors",
            "Velocity Sensors": "Velocity Sensors",
            "Speed Sensors": "Speed Sensors",
            "Acceleration Sensors": "Acceleration Sensors",
            "Jerk Sensors": "Jerk Sensors",
            "Attitude Sensors": "Attitude Sensors",
            "Orientation Sensors": "Orientation Sensors",
            "Heading Sensors": "Heading Sensors",
            "Tilt Sensors": "Tilt Sensors",
            "Level Sensors": "Level Sensors",
            "Flow Sensors": "Flow Sensors",
            "Liquid Flow Sensors": "Liquid Flow Sensors",
            "Gas Flow Sensors": "Gas Flow Sensors",
            "Mass Flow Sensors": "Mass Flow Sensors",
            "Volumetric Flow Sensors": "Volumetric Flow Sensors",
            "Level Sensors": "Level Sensors",
            "Liquid Level Sensors": "Liquid Level Sensors",
            "Solid Level Sensors": "Solid Level Sensors",
            "Fill Level Sensors": "Fill Level Sensors",
            "Distance Sensors": "Distance Sensors",
            "Range Sensors": "Range Sensors",
            "Proximity Sensors": "Proximity Sensors",
            "Collision Avoidance Sensors": "Collision Avoidance Sensors",
            "Object Detection Sensors": "Object Detection Sensors",
            "Object Recognition Sensors": "Object Recognition Sensors",
            "Face Detection Sensors": "Face Detection Sensors",
            "Face Recognition Sensors": "Face Recognition Sensors",
            "Gesture Recognition Sensors": "Gesture Recognition Sensors",
            "Voice Recognition Sensors": "Voice Recognition Sensors",
            "Speech Recognition Sensors": "Speech Recognition Sensors",
            "Sound Recognition Sensors": "Sound Recognition Sensors",
            "Image Recognition Sensors": "Image Recognition Sensors",
            "Pattern Recognition Sensors": "Pattern Recognition Sensors",
            "Optical Sensors": "Optical Sensors",
            "Laser Sensors": "Laser Sensors",
            "Fiber Optic Sensors": "Fiber Optic Sensors",
            "Photodetectors": "Photodetectors",
            "Photodiodes": "Photodiodes",
            "Phototransistors": "Phototransistors",
            "Solar Cells": "Solar Cells",
            "Photovoltaic Cells": "Photovoltaic Cells",
            "Thermal Sensors": "Thermal Sensors",
            "Infrared Sensors": "Infrared Sensors",
            "Thermocouples": "Thermocouples",
            "Resistance Temperature Detectors": "Resistance Temperature Detectors",
            "Thermistors": "Thermistors",
            "Infrared Thermometers": "Infrared Thermometers",
            "Thermal Imaging Cameras": "Thermal Imaging Cameras",
            "Heat Flux Sensors": "Heat Flux Sensors",
            "Chemical Sensors": "Chemical Sensors",
            "Gas Sensors": "Gas Sensors",
            "Liquid Sensors": "Liquid Sensors",
            "pH Sensors": "pH Sensors",
            "Conductivity Sensors": "Conductivity Sensors",
            "Oxygen Sensors": "Oxygen Sensors",
            "Carbon Dioxide Sensors": "Carbon Dioxide Sensors",
            "Methane Sensors": "Methane Sensors",
            "Hydrogen Sensors": "Hydrogen Sensors",
            "Nitrogen Sensors": "Nitrogen Sensors",
            "Sulfur Sensors": "Sulfur Sensors",
            "Heavy Metal Sensors": "Heavy Metal Sensors",
            "Pollution Sensors": "Pollution Sensors",
            "Biosensors": "Biosensors",
            "Medical Sensors": "Medical Sensors",
            "Health Sensors": "Health Sensors",
            "Biomedical Sensors": "Biomedical Sensors",
            "Clinical Sensors": "Clinical Sensors",
            "Point-of-Care Sensors": "Point-of-Care Sensors",
            "Diagnostic Sensors": "Diagnostic Sensors",
            "Monitoring Sensors": "Monitoring Sensors",
            "Wearable Medical Sensors": "Wearable Medical Sensors",
            "Implantable Medical Sensors": "Implantable Medical Sensors",
            "Ingestible Medical Sensors": "Ingestible Medical Sensors",
            "Non-invasive Medical Sensors": "Non-invasive Medical Sensors",
            "Invasive Medical Sensors": "Invasive Medical Sensors",
            "Contact Medical Sensors": "Contact Medical Sensors",
            "Non-contact Medical Sensors": "Non-contact Medical Sensors",
            "Wireless Medical Sensors": "Wireless Medical Sensors",
            "Wired Medical Sensors": "Wired Medical Sensors",
            "Passive Medical Sensors": "Passive Medical Sensors",
            "Active Medical Sensors": "Active Medical Sensors",
            "Digital Medical Sensors": "Digital Medical Sensors",
            "Analog Medical Sensors": "Analog Medical Sensors",
            "MEMS Medical Sensors": "MEMS Medical Sensors",
            "NEMS Medical Sensors": "NEMS Medical Sensors",
            "CMOS Medical Sensors": "CMOS Medical Sensors",
            "CCD Medical Sensors": "CCD Medical Sensors",
            "Image Medical Sensors": "Image Medical Sensors",
            "Vision Medical Sensors": "Vision Medical Sensors",
            "Camera Medical Sensors": "Camera Medical Sensors",
            "Video Medical Sensors": "Video Medical Sensors",
            "Audio Medical Sensors": "Audio Medical Sensors",
            "Microphone Medical Sensors": "Microphone Medical Sensors",
            "Sound Medical Sensors": "Sound Medical Sensors",
            "Vibration Medical Sensors": "Vibration Medical Sensors",
            "Accelerometer Medical Sensors": "Accelerometer Medical Sensors",
            "Gyroscope Medical Sensors": "Gyroscope Medical Sensors",
            "Magnetometer Medical Sensors": "Magnetometer Medical Sensors",
            "Inertial Medical Sensors": "Inertial Medical Sensors",
            "GPS Medical Sensors": "GPS Medical Sensors",
            "GNSS Medical Sensors": "GNSS Medical Sensors",
            "Navigation Medical Sensors": "Navigation Medical Sensors",
            "Position Medical Sensors": "Position Medical Sensors",
            "Location Medical Sensors": "Location Medical Sensors",
            "Tracking Medical Sensors": "Tracking Medical Sensors",
            "Motion Medical Sensors": "Motion Medical Sensors",
            "Velocity Medical Sensors": "Velocity Medical Sensors",
            "Speed Medical Sensors": "Speed Medical Sensors",
            "Acceleration Medical Sensors": "Acceleration Medical Sensors",
            "Jerk Medical Sensors": "Jerk Medical Sensors",
            "Attitude Medical Sensors": "Attitude Medical Sensors",
            "Orientation Medical Sensors": "Orientation Medical Sensors",
            "Heading Medical Sensors": "Heading Medical Sensors",
            "Tilt Medical Sensors": "Tilt Medical Sensors",
            "Level Medical Sensors": "Level Medical Sensors",
            "Flow Medical Sensors": "Flow Medical Sensors",
            "Liquid Flow Medical Sensors": "Liquid Flow Medical Sensors",
            "Gas Flow Medical Sensors": "Gas Flow Medical Sensors",
            "Mass Flow Medical Sensors": "Mass Flow Medical Sensors",
            "Volumetric Flow Medical Sensors": "Volumetric Flow Medical Sensors",
            "Level Medical Sensors": "Level Medical Sensors",
            "Liquid Level Medical Sensors": "Liquid Level Medical Sensors",
            "Solid Level Medical Sensors": "Solid Level Medical Sensors",
            "Fill Level Medical Sensors": "Fill Level Medical Sensors",
            "Distance Medical Sensors": "Distance Medical Sensors",
            "Range Medical Sensors": "Range Medical Sensors",
            "Proximity Medical Sensors": "Proximity Medical Sensors",
            "Collision Avoidance Medical Sensors": "Collision Avoidance Medical Sensors",
            "Object Detection Medical Sensors": "Object Detection Medical Sensors",
            "Object Recognition Medical Sensors": "Object Recognition Medical Sensors",
            "Face Detection Medical Sensors": "Face Detection Medical Sensors",
            "Face Recognition Medical Sensors": "Face Recognition Medical Sensors",
            "Gesture Recognition Medical Sensors": "Gesture Recognition Medical Sensors",
            "Voice Recognition Medical Sensors": "Voice Recognition Medical Sensors",
            "Speech Recognition Medical Sensors": "Speech Recognition Medical Sensors",
            "Sound Recognition Medical Sensors": "Sound Recognition Medical Sensors",
            "Image Recognition Medical Sensors": "Image Recognition Medical Sensors",
            "Pattern Recognition Medical Sensors": "Pattern Recognition Medical Sensors",
            "Optical Medical Sensors": "Optical Medical Sensors",
            "Laser Medical Sensors": "Laser Medical Sensors",
            "Fiber Optic Medical Sensors": "Fiber Optic Medical Sensors",
            "Photodetector Medical Sensors": "Photodetector Medical Sensors",
            "Photodiode Medical Sensors": "Photodiode Medical Sensors",
            "Phototransistor Medical Sensors": "Phototransistor Medical Sensors",
            "Solar Cell Medical Sensors": "Solar Cell Medical Sensors",
            "Photovoltaic Cell Medical Sensors": "Photovoltaic Cell Medical Sensors",
            "Thermal Medical Sensors": "Thermal Medical Sensors",
            "Infrared Medical Sensors": "Infrared Medical Sensors",
            "Thermocouple Medical Sensors": "Thermocouple Medical Sensors",
            "Resistance Temperature Detector Medical Sensors": "Resistance Temperature Detector Medical Sensors",
            "Thermistor Medical Sensors": "Thermistor Medical Sensors",
            "Infrared Thermometer Medical Sensors": "Infrared Thermometer Medical Sensors",
            "Thermal Imaging Camera Medical Sensors": "Thermal Imaging Camera Medical Sensors",
            "Heat Flux Medical Sensors": "Heat Flux Medical Sensors",
            "Chemical Medical Sensors": "Chemical Medical Sensors",
            "Gas Medical Sensors": "Gas Medical Sensors",
            "Liquid Medical Sensors": "Liquid Medical Sensors",
            "pH Medical Sensors": "pH Medical Sensors",
            "Conductivity Medical Sensors": "Conductivity Medical Sensors",
            "Oxygen Medical Sensors": "Oxygen Medical Sensors",
            "Carbon Dioxide Medical Sensors": "Carbon Dioxide Medical Sensors",
            "Methane Medical Sensors": "Methane Medical Sensors",
            "Hydrogen Medical Sensors": "Hydrogen Medical Sensors",
            "Nitrogen Medical Sensors": "Nitrogen Medical Sensors",
            "Sulfur Medical Sensors": "Sulfur Medical Sensors",
            "Heavy Metal Medical Sensors": "Heavy Metal Medical Sensors",
            "Pollution Medical Sensors": "Pollution Medical Sensors"
        }
        
        expanded_query = query
        for abbr, full in abbreviations.items():
            if f" {abbr} " in expanded_query or expanded_query.startswith(abbr) or expanded_query.endswith(abbr):
                expanded_query = expanded_query.replace(f" {abbr} ", f" {abbr} OR {full} ")
                if expanded_query.startswith(abbr):
                    expanded_query = f"{abbr} OR {full}{expanded_query[len(abbr):]}"
                if expanded_query.endswith(abbr):
                    expanded_query = f"{expanded_query[:-len(abbr)]}{abbr} OR {full}"
        
        return expanded_query

if __name__ == "__main__":
    # Test the QueryExpander
    expander = QueryExpander()
    
    test_queries = [
        "CVPR VLA 自动驾驶",
        "Mamba architecture for vision",
        "2024 NIPS transformer models for protein folding"
    ]
    
    for query in test_queries:
        print(f"Original query: {query}")
        expanded = expander.expand_query(query)
        print(f"Expanded query: {json.dumps(expanded, indent=2, ensure_ascii=False)}")
        print()
