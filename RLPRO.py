import os
import random
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, Any
from groq import Groq
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define Groq API Key directly in code
GROQ_API_KEY = "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"

# Configuration
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.3
EXPLORATION_DECAY = 0.99

# Environment
class AdvertisingEnvironment:
    def __init__(self):
        self.business_types = [
            "Clothing Store", "Coffee Shop", "Restaurant", "Fitness Center",
            "Language School", "Electronics Store", "Beauty Salon",
            "Travel Agency", "Home Appliance Store", "Bookstore",
            "E-commerce Platform", "SaaS Business", "Mobile App", "Real Estate Agency",
            "Consulting Firm", "Healthcare Service", "Financial Service"
        ]
        self.target_audiences = [
            "Young Adults 18-25", "Teenagers 13-17", "Adults 26-35",
            "Middle-aged 36-50", "Seniors 50+", "Women", "Men",
            "College Students", "Athletes", "Tech Enthusiasts", "Parents",
            "Professionals", "Business Owners", "International Travelers",
            "Luxury Consumers", "Budget Shoppers", "Remote Workers"
        ]
        self.ad_types = [
            "Short Video", "Instagram Post", "Instagram Story",
            "Website Banner", "LinkedIn Post", "Text Ad",
            "YouTube Ad", "Google Display Ad", "Facebook Carousel",
            "Email Newsletter", "Influencer Collaboration", "TikTok Video"
        ]
        self.ad_goals = [
            "Brand Awareness", "Lead Generation", "Sales Conversion",
            "Website Traffic", "App Downloads", "Engagement",
            "Customer Retention", "Product Launch"
        ]
        self.display_times = [
            "Morning", "Afternoon", "Evening", "Late Night",
            "Weekday", "Weekend", "Business Hours", "After Hours"
        ]
        self.reset()
        self.click_probabilities = self._initialize_click_probabilities()

    def _initialize_click_probabilities(self) -> Dict:
        probabilities = {}
        for business in self.business_types:
            probabilities[business] = {}
            for audience in self.target_audiences:
                probabilities[business][audience] = {}
                for ad_type in self.ad_types:
                    probabilities[business][audience][ad_type] = {}
                    for time in self.display_times:
                        base_prob = random.uniform(0.01, 0.15)
                        business_audience_match = self._get_business_audience_match(business, audience)
                        ad_type_audience_match = self._get_ad_type_audience_match(ad_type, audience)
                        time_audience_match = self._get_time_audience_match(time, audience)
                        final_prob = base_prob * business_audience_match * ad_type_audience_match * time_audience_match
                        final_prob = min(max(final_prob, 0.005), 0.3)
                        probabilities[business][audience][ad_type][time] = final_prob
        return probabilities

    def _get_business_audience_match(self, business: str, audience: str) -> float:
        good_matches = {
            "Clothing Store": ["Young Adults 18-25", "Teenagers 13-17", "Women"],
            "Coffee Shop": ["Young Adults 18-25", "Professionals", "College Students"],
            "Restaurant": ["Adults 26-35", "Young Adults 18-25", "Luxury Consumers"],
            "Fitness Center": ["Adults 26-35", "Athletes", "Young Adults 18-25"],
            "Language School": ["College Students", "Young Adults 18-25", "Professionals"],
            "Electronics Store": ["Tech Enthusiasts", "Young Adults 18-25", "Men"],
            "Beauty Salon": ["Women", "Adults 26-35", "Young Adults 18-25"],
            "Travel Agency": ["International Travelers", "Adults 26-35", "Luxury Consumers"],
            "E-commerce Platform": ["Adults 26-35", "Remote Workers", "Tech Enthusiasts"],
            "SaaS Business": ["Business Owners", "Professionals", "Tech Enthusiasts"],
            "Financial Service": ["Professionals", "Business Owners", "Adults 26-35"]
        }
        if business in good_matches and audience in good_matches[business]:
            return random.uniform(1.5, 2.5)
        return random.uniform(0.7, 1.3)

    def _get_ad_type_audience_match(self, ad_type: str, audience: str) -> float:
        good_matches = {
            "Instagram Post": ["Young Adults 18-25", "Teenagers 13-17"],
            "Instagram Story": ["Young Adults 18-25", "Teenagers 13-17"],
            "LinkedIn Post": ["Professionals", "Business Owners"],
            "TikTok Video": ["Teenagers 13-17", "Young Adults 18-25"],
            "YouTube Ad": ["Tech Enthusiasts", "Young Adults 18-25"],
            "Email Newsletter": ["Professionals", "Business Owners", "Adults 26-35"]
        }
        if ad_type in good_matches and audience in good_matches[ad_type]:
            return random.uniform(1.5, 2.5)
        return random.uniform(0.7, 1.3)

    def _get_time_audience_match(self, time: str, audience: str) -> float:
        good_matches = {
            "Morning": ["Professionals", "Adults 26-35"],
            "Evening": ["Young Adults 18-25", "Adults 26-35"],
            "Late Night": ["Young Adults 18-25", "Teenagers 13-17"],
            "Weekend": ["Teenagers 13-17", "International Travelers"],
            "Business Hours": ["Professionals", "Business Owners"],
            "After Hours": ["Young Adults 18-25", "College Students"]
        }
        if time in good_matches and audience in good_matches[time]:
            return random.uniform(1.5, 2.0)
        return random.uniform(0.8, 1.2)

    def reset(self):
        self.current_business = None
        self.current_audience = None
        self.current_ad_type = None
        self.current_goal = None
        self.current_display_time = None
        self.total_reward = 0
        self.actions_taken = 0
        self.clicks = 0
        self.impressions = 0
        return self._get_state()

    def _get_state(self) -> Dict:
        return {
            "business_type": self.current_business,
            "target_audience": self.current_audience,
            "ad_type": self.current_ad_type,
            "ad_goal": self.current_goal,
            "display_time": self.current_display_time,
            "ctr": self.clicks / max(1, self.impressions)
        }

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        self.current_business = action.get("business_type", self.current_business)
        self.current_audience = action.get("target_audience", self.current_audience)
        self.current_ad_type = action.get("ad_type", self.current_ad_type)
        self.current_goal = action.get("ad_goal", self.current_goal)
        self.current_display_time = action.get("display_time", self.current_display_time)
        reward = 0
        done = False
        info = {}
        if (self.current_business and self.current_audience and
                self.current_ad_type and self.current_display_time):
            self.impressions += 1
            click_probability = self.click_probabilities[self.current_business][self.current_audience][self.current_ad_type][self.current_display_time]
            click_probability *= random.uniform(0.8, 1.2)
            clicked = random.random() < click_probability
            if clicked:
                self.clicks += 1
                reward = 1.0
            else:
                reward = -0.1
            self.total_reward += reward
            self.actions_taken += 1
            info["click_probability"] = click_probability
            info["clicked"] = clicked
            info["ctr"] = self.clicks / self.impressions
            if self.actions_taken >= 100:
                done = True
        return self._get_state(), reward, done, info

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.exploration_rate = EXPLORATION_RATE
        self.exploration_decay = EXPLORATION_DECAY
        self.q_table = {}
        self.business_types = env.business_types
        self.target_audiences = env.target_audiences
        self.ad_types = env.ad_types
        self.display_times = env.display_times

    def _get_state_key(self, state):
        return (
            state["business_type"],
            state["target_audience"],
            state["ad_type"],
            state["display_time"]
        )

    def _get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        action_key = (
            action["business_type"],
            action["target_audience"],
            action["ad_type"],
            action["display_time"]
        )
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        return self.q_table[state_key][action_key]

    def _set_q_value(self, state, action, value):
        state_key = self._get_state_key(state)
        action_key = (
            action["business_type"],
            action["target_audience"],
            action["ad_type"],
            action["display_time"]
        )
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = value

    def choose_action(self, state, business_type=None):
        if random.random() < self.exploration_rate:
            return {
                "business_type": business_type if business_type else random.choice(self.business_types),
                "target_audience": random.choice(self.target_audiences),
                "ad_type": random.choice(self.ad_types),
                "display_time": random.choice(self.display_times)
            }
        else:
            state_key = self._get_state_key(state)
            if state_key not in self.q_table or not self.q_table[state_key]:
                return {
                    "business_type": business_type if business_type else random.choice(self.business_types),
                    "target_audience": random.choice(self.target_audiences),
                    "ad_type": random.choice(self.ad_types),
                    "display_time": random.choice(self.display_times)
                }
            best_action_key = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return {
                "business_type": best_action_key[0],
                "target_audience": best_action_key[1],
                "ad_type": best_action_key[2],
                "display_time": best_action_key[3]
            }

    def learn(self, state, action, reward, next_state):
        current_q = self._get_q_value(state, action)
        max_next_q = 0.0
        state_key = self._get_state_key(next_state)
        if state_key in self.q_table and self.q_table[state_key]:
            max_next_q = max(self.q_table[state_key].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self._set_q_value(state, action, new_q)
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)

# Groq Ad Content Generator
class GroqAdContentGenerator:
    def __init__(self):
        self.client = None
        self.model = GROQ_MODEL
        try:
            self.client = Groq(api_key=GROQ_API_KEY)
            logging.info("Groq API client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Groq API client: {str(e)}")
            self.client = None

    def generate_ad_content(self, business_type, target_audience=None, ad_type=None, ad_goal=None):
        if not self.client:
            logging.warning("Groq client is not initialized")
            return {
                "headline": "Connect your Groq API key for AI-generated content",
                "body": "This is a placeholder. Connect your Groq API to get AI-generated ad content.",
                "cta": "Learn More"
            }
        try:
            prompt = f"""
            Create an ad for a {business_type}{f" targeting {target_audience}" if target_audience else ""}{f" as a {ad_type}" if ad_type else ""}{f" to achieve {ad_goal}" if ad_goal else ""}.
            Return JSON with: headline (up to 10 words), body (up to 30 words), cta (up to 3 words).
            """
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1000,
                top_p=1,
                stream=True,
                stop=None
            )
            response_text = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                response_text += content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                ad_content = json.loads(json_text)
                return ad_content
            else:
                lines = response_text.strip().split('\n')
                ad_content = {}
                for line in lines:
                    if "headline" in line.lower():
                        ad_content["headline"] = line.split(":", 1)[1].strip()
                    elif "body" in line.lower():
                        ad_content["body"] = line.split(":", 1)[1].strip()
                    elif "cta" in line.lower():
                        ad_content["cta"] = line.split(":", 1)[1].strip()
                return ad_content
        except Exception as e:
            logging.error(f"Error in generate_ad_content: {str(e)}")
            return {
                "headline": "Discover the Difference Today",
                "body": "Experience what makes us unique. Our products are designed with you in mind.",
                "cta": "Shop Now"
            }

    def generate_ad_suggestions(self, business_type):
        if not self.client:
            logging.warning("Groq client is not initialized")
            return {
                "target_audiences": ["Young Adults 18-25", "Professionals"],
                "ad_types": ["Instagram Post", "Website Banner"],
                "display_times": ["Evening", "Weekend"],
                "ad_goal": "Brand Awareness"
            }
        try:
            prompt = f"""
            Suggest an ad strategy for a {business_type}.
            Return JSON with: target_audiences (2-3 segments), ad_types (2-3 formats), display_times (2-3 times), ad_goal (1 goal).
            """
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1000,
                top_p=1,
                stream=True,
                stop=None
            )
            response_text = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                response_text += content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                suggestions = json.loads(json_text)
                return suggestions
            else:
                return {
                    "target_audiences": ["Young Adults 18-25", "Professionals"],
                    "ad_types": ["Instagram Post", "Website Banner"],
                    "display_times": ["Evening", "Weekend"],
                    "ad_goal": "Brand Awareness"
                }
        except Exception as e:
            logging.error(f"Error in generate_ad_suggestions: {str(e)}")
            return {
                "target_audiences": ["Young Adults 18-25", "Professionals"],
                "ad_types": ["Instagram Post", "Website Banner"],
                "display_times": ["Evening", "Weekend"],
                "ad_goal": "Brand Awareness"
            }

    def interact_with_model(self, user_input):
        if not self.client:
            logging.warning("Groq client is not initialized")
            return "Error: Unable to initialize Groq client. Please check the API key."
        try:
            prompt = f"""
            You are a friendly advertising assistant. The user said: "{user_input}". Respond naturally:
            - For greetings (e.g., "hi"), reply warmly and ask about their ad needs.
            - For ad requests, return JSON with headline, body, cta.
            - For strategy suggestions, return JSON with target_audiences, ad_types, display_times, ad_goal.
            - For other queries, give a short text reply (up to 100 words) with a follow-up question.
            """
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1000,
                top_p=1,
                stream=True,
                stop=None
            )
            response_text = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content or ""
                response_text += content
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                return json.loads(json_text)
            return response_text.strip()
        except Exception as e:
            logging.error(f"Error in interact_with_model: {str(e)}")
            return f"Error: Unable to interact with the model. Details: {str(e)}"

# Training
def train_agent(num_episodes=2000):
    env = AdvertisingEnvironment()
    agent = QLearningAgent(env)
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
    return agent, env

# UI Application
class AdOptimizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AdOptimizer: RL-Powered Ad Platform")
        self.root.geometry("1280x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.env = AdvertisingEnvironment()
        self.agent, _ = train_agent()
        self.content_generator = GroqAdContentGenerator()
        self.suggestions = None
        self.ad_content = None
        self.campaign_results = []
        self.history = []

        self.setup_ui()

    def setup_ui(self):
        self.tab_view = ctk.CTkTabview(self.root, fg_color="#1C2526")
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Campaign Optimization
        self.campaign_tab = self.tab_view.add("Campaign Optimization")
        self.setup_campaign_tab()

        # Tab 2: Model Interaction
        self.model_tab = self.tab_view.add("Model Interaction")
        self.setup_model_tab()

    def setup_campaign_tab(self):
        main_frame = ctk.CTkFrame(self.campaign_tab, fg_color="#1C2526")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Panel: Business Selection
        self.left_panel = ctk.CTkFrame(main_frame, fg_color="#2D383A", width=350)
        self.left_panel.pack(side="left", fill="y", padx=5, pady=5)

        ctk.CTkLabel(self.left_panel, text="Select Business Type", font=("Roboto", 18, "bold")).pack(pady=15)
        self.business_var = ctk.StringVar()
        self.business_dropdown = ctk.CTkOptionMenu(self.left_panel, variable=self.business_var, values=self.env.business_types, fg_color="#3A4F50", button_color="#4A6366", font=("Roboto", 14), width=300, height=40)
        self.business_dropdown.pack(pady=15)

        self.generate_button = ctk.CTkButton(self.left_panel, text="Generate Recommendations", command=self.generate_recommendations, fg_color="#4A6366", hover_color="#5A7376", font=("Roboto", 14), width=300, height=40)
        self.generate_button.pack(pady=15)

        # Right Panel: Campaign Customization and Preview
        self.right_panel = ctk.CTkFrame(main_frame, fg_color="#2D383A")
        self.right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Campaign Customization
        self.customize_frame = ctk.CTkFrame(self.right_panel, fg_color="#3A4F50")
        self.customize_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.customize_frame, text="Customize Campaign", font=("Roboto", 16, "bold")).pack(pady=10)

        self.audience_var = ctk.StringVar()
        self.ad_type_var = ctk.StringVar()
        self.time_var = ctk.StringVar()
        self.goal_var = ctk.StringVar()

        self.audience_dropdown = ctk.CTkOptionMenu(self.customize_frame, variable=self.audience_var, values=self.env.target_audiences, fg_color="#4A6366", button_color="#5A7376", font=("Roboto", 12), width=200, height=35)
        self.audience_dropdown.pack(side="left", padx=5, pady=5)

        self.ad_type_dropdown = ctk.CTkOptionMenu(self.customize_frame, variable=self.ad_type_var, values=self.env.ad_types, fg_color="#4A6366", button_color="#5A7376", font=("Roboto", 12), width=200, height=35)
        self.ad_type_dropdown.pack(side="left", padx=5, pady=5)

        self.time_dropdown = ctk.CTkOptionMenu(self.customize_frame, variable=self.time_var, values=self.env.display_times, fg_color="#4A6366", button_color="#5A7376", font=("Roboto", 12), width=200, height=35)
        self.time_dropdown.pack(side="left", padx=5, pady=5)

        self.goal_dropdown = ctk.CTkOptionMenu(self.customize_frame, variable=self.goal_var, values=self.env.ad_goals, fg_color="#4A6366", button_color="#5A7376", font=("Roboto", 12), width=200, height=35)
        self.goal_dropdown.pack(side="left", padx=5, pady=5)

        self.refresh_button = ctk.CTkButton(self.customize_frame, text="Refresh Ad Content", command=self.refresh_ad_content, fg_color="#4A6366", hover_color="#5A7376", font=("Roboto", 12), width=200, height=35)
        self.refresh_button.pack(side="left", padx=5, pady=5)

        # Ad Preview and Settings
        self.preview_frame = ctk.CTkFrame(self.right_panel, fg_color="#3A4F50")
        self.preview_frame.pack(fill="x", padx=10, pady=10)

        self.preview_label = ctk.CTkLabel(self.preview_frame, text="Ad Preview", font=("Roboto", 16, "bold"))
        self.preview_label.pack(pady=10)

        self.preview_text = ctk.CTkTextbox(self.preview_frame, height=150, fg_color="#2D383A", font=("Roboto", 14))
        self.preview_text.pack(fill="x", padx=5, pady=5)

        self.settings_text = ctk.CTkTextbox(self.preview_frame, height=150, fg_color="#2D383A", font=("Roboto", 14))
        self.settings_text.pack(fill="x", padx=5, pady=5)

        self.simulate_button = ctk.CTkButton(self.preview_frame, text="Run Campaign Simulation", command=self.run_simulation, fg_color="#4A6366", hover_color="#5A7376", font=("Roboto", 14), width=300, height=40)
        self.simulate_button.pack(pady=15)

        # Results
        self.results_frame = ctk.CTkFrame(self.right_panel, fg_color="#3A4F50")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.results_label = ctk.CTkLabel(self.results_frame, text="Campaign Results", font=("Roboto", 16, "bold"))
        self.results_label.pack(pady=10)

        self.results_text = ctk.CTkTextbox(self.results_frame, height=200, fg_color="#2D383A", font=("Roboto", 14))
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_model_tab(self):
        model_frame = ctk.CTkFrame(self.model_tab, fg_color="#1C2526")
        model_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Input Section
        input_frame = ctk.CTkFrame(model_frame, fg_color="#2D383A")
        input_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(input_frame, text="Talk to the Advertising Assistant", font=("Roboto", 18, "bold")).pack(pady=10)

        ctk.CTkLabel(input_frame, text="Your Message", font=("Roboto", 14)).pack(pady=5)
        self.model_query = ctk.CTkTextbox(input_frame, height=100, fg_color="#3A4F50", font=("Roboto", 12))
        self.model_query.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(input_frame, text="Send Message", command=self.submit_model_query, fg_color="#4A6366", hover_color="#5A7376", font=("Roboto", 14), width=300, height=40).pack(pady=15)

        # Output Section
        output_frame = ctk.CTkFrame(model_frame, fg_color="#2D383A")
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(output_frame, text="Assistant Response", font=("Roboto", 16, "bold")).pack(pady=10)
        self.model_response = ctk.CTkTextbox(output_frame, height=300, fg_color="#3A4F50", font=("Roboto", 14))
        self.model_response.pack(fill="both", expand=True, padx=5, pady=5)

    def generate_recommendations(self):
        business_type = self.business_var.get()
        if not business_type:
            messagebox.showwarning("Warning", "Please select a business type!")
            return

        self.suggestions = self.content_generator.generate_ad_suggestions(business_type)
        state = self.env.reset()
        action = self.agent.choose_action(state, business_type=business_type)

        if not self.suggestions.get("target_audiences"):
            self.suggestions["target_audiences"] = [action["target_audience"]]
        if not self.suggestions.get("ad_types"):
            self.suggestions["ad_types"] = [action["ad_type"]]
        if not self.suggestions.get("display_times"):
            self.suggestions["display_times"] = [action["display_time"]]

        self.ad_content = self.content_generator.generate_ad_content(
            business_type,
            target_audience=self.suggestions["target_audiences"][0],
            ad_type=self.suggestions["ad_types"][0],
            ad_goal=self.suggestions["ad_goal"]
        )

        self.audience_var.set(self.suggestions["target_audiences"][0])
        self.ad_type_var.set(self.suggestions["ad_types"][0])
        self.time_var.set(self.suggestions["display_times"][0])
        self.goal_var.set(self.suggestions["ad_goal"])

        self.update_preview()

    def refresh_ad_content(self):
        if not self.business_var.get():
            messagebox.showwarning("Warning", "Please select a business type!")
            return

        self.ad_content = self.content_generator.generate_ad_content(
            self.business_var.get(),
            target_audience=self.audience_var.get(),
            ad_type=self.ad_type_var.get(),
            ad_goal=self.goal_var.get()
        )
        self.update_preview()

    def update_preview(self):
        self.preview_text.delete("1.0", tk.END)
        self.settings_text.delete("1.0", tk.END)

        if self.ad_content:
            preview_content = f"{self.ad_content.get('headline', 'Compelling Headline')}\n\n"
            preview_content += f"{self.ad_content.get('body', 'Ad body text goes here.')}\n\n"
            preview_content += f"CTA: {self.ad_content.get('cta', 'Click Here')}"
            self.preview_text.insert("1.0", preview_content)

            settings_content = f"Business Type: {self.business_var.get()}\n"
            settings_content += f"Target Audience: {self.audience_var.get()}\n"
            settings_content += f"Ad Type: {self.ad_type_var.get()}\n"
            settings_content += f"Display Time: {self.time_var.get()}\n"
            settings_content += f"Campaign Goal: {self.goal_var.get()}"
            self.settings_text.insert("1.0", settings_content)

    def run_simulation(self):
        if not all([self.business_var.get(), self.audience_var.get(), self.ad_type_var.get(), self.time_var.get(), self.goal_var.get()]):
            messagebox.showwarning("Warning", "Please complete all campaign settings!")
            return

        self.env.reset()
        state = {
            "business_type": self.business_var.get(),
            "target_audience": self.audience_var.get(),
            "ad_type": self.audience_var.get(),
            "display_time": self.time_var.get(),
            "ad_goal": self.goal_var.get(),
            "ctr": 0
        }
        action = {
            "business_type": self.business_var.get(),
            "target_audience": self.audience_var.get(),
            "ad_type": self.ad_type_var.get(),
            "display_time": self.time_var.get(),
            "ad_goal": self.goal_var.get()
        }
        impressions = 1000
        clicks = 0
        total_reward = 0
        for _ in range(impressions):
            _, reward, _, info = self.env.step(action)
            total_reward += reward
            if info.get("clicked", False):
                clicks += 1
        ctr = clicks / impressions
        result = {
            "business_type": self.business_var.get(),
            "target_audience": self.audience_var.get(),
            "ad_type": self.ad_type_var.get(),
            "display_time": self.time_var.get(),
            "ad_goal": self.goal_var.get(),
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "headline": self.ad_content.get('headline', ''),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.campaign_results.append(result)
        self.history.append(result)
        self.agent.learn(state, action, total_reward, state)

        self.results_text.delete("1.0", tk.END)
        results_content = f"Latest Campaign Performance\n\n"
        results_content += f"Impressions: {result['impressions']:,}\n"
        results_content += f"Clicks: {result['clicks']:,}\n"
        results_content += f"CTR: {result['ctr']*100:.2f}%\n"
        results_content += f"Headline: {result['headline']}\n"
        results_content += f"Targeting: {result['target_audience']}\n"
        results_content += f"Ad Format: {result['ad_type']}\n"
        results_content += f"Time: {result['display_time']}\n\n"
        results_content += "Campaign History\n"
        for hist in self.history[-3:]:
            results_content += f"{hist['timestamp']} | {hist['business_type']} | {hist['ad_type']} | Clicks: {hist['clicks']} | CTR: {hist['ctr']*100:.2f}%\n"
        self.results_text.insert("1.0", results_content)

    def submit_model_query(self):
        user_input = self.model_query.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Warning", "Please enter a message!")
            return

        response = self.content_generator.interact_with_model(user_input)

        self.model_response.delete("1.0", tk.END)
        if isinstance(response, dict):
            self.model_response.insert("1.0", json.dumps(response, indent=2))
        else:
            self.model_response.insert("1.0", response)

if __name__ == "__main__":
    root = ctk.CTk()
    app = AdOptimizerApp(root)
    root.mainloop()