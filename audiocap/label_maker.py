import re


class LabelMaker():
    def __init__(self):
        pass
      
    # --- Mapping Functions ---
    @staticmethod
    def translate_emotion(value):
        if value == 0:
            return "Not Present At All"
        elif value == 1:
            return "Slightly Present"
        elif value == 2:
            return "Moderately Present"
        elif value == 3:
            return "Strongly Present"
        elif value == 4:
            return "Extremely Present"
        else:
            return "Unknown"

    @staticmethod
    def translate_dimension(dimension_name, value):
        if dimension_name == "Valence":
            if value == -3:
                return "Extremely Negative"
            elif value == -2:
                return "Very Negative"
            elif value == -1:
                return "Slightly Negative"
            elif value == 0:
                return "Neutral"
            elif value == 1:
                return "Slightly Positive"
            elif value == 2:
                return "Very Positive"
            elif value == 3:
                return "Extremely Positive"
            else:
                return "Unknown"

        elif dimension_name == "Arousal":
            if value == 0:
                return "Very Calm"
            elif value == 1:
                return "Slightly Calm"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Excited"
            elif value == 4:
                return "Very Excited"
            else:
                return "Unknown"

        elif dimension_name == "Submissive vs. Dominant":
            if value == -3:
                return "Extremely Submissive"
            elif value == -2:
                return "Very Submissive"
            elif value == -1:
                return "Slightly Submissive"
            elif value == 0:
                return "Neutral"
            elif value == 1:
                return "Slightly Dominant"
            elif value == 2:
                return "Very Dominant"
            elif value == 3:
                return "Extremely Dominant"
            else:
                return "Unknown"

        elif dimension_name == "Age":
            if value == 0:
                return "Infant/Toddler"
            elif value == 1:
                return "Little Kid"
            elif value == 2:
                return "Teenager"
            elif value == 3:
                return "Young Adult"
            elif value == 4:
                return "Adult"
            elif value == 5:
                return "Elderly"
            elif value == 6:
                return "Very Old"
            else:
                return "Unknown"

        elif dimension_name == "Gender":
            if value == -2:
                return "Clearly/Very Masculine"
            elif value == -1:
                return "Somewhat Masculine"
            elif value == 0:
                return "Neutral/Unsure"
            elif value == 1:
                return "Somewhat Feminine"
            elif value == 2:
                return "Clearly/Very Feminine"
            else:
                return "Unknown"

        elif dimension_name == "Serious vs. Humorous":
            if value == 0:
                return "Very Serious"
            elif value == 1:
                return "Slightly Serious"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Humorous"
            elif value == 4:
                return "Very Humorous"
            else:
                return "Unknown"

        elif dimension_name == "Vulnerable vs. Emotionally Detached":
            if value == 0:
                return "Very Vulnerable"
            elif value == 1:
                return "Slightly Vulnerable"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Detached"
            elif value == 4:
                return "Very Detached"
            else:
                return "Unknown"

        elif dimension_name == "Confident vs. Hesitant":
            if value == 0:
                return "Very Confident"
            elif value == 1:
                return "Slightly Confident"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Hesitant"
            elif value == 4:
                return "Very Hesitant"
            else:
                return "Unknown"

        elif dimension_name == "Warm vs. Cold":
            if value == -2:
                return "Very Cold"
            elif value == -1:
                return "Slightly Cold"
            elif value == 0:
                return "Neutral"
            elif value == 1:
                return "Slightly Warm"
            elif value == 2:
                return "Very Warm"
            else:
                return "Unknown"

        elif dimension_name == "Monotone vs. Expressive":
            if value == 0:
                return "Very Monotone"
            elif value == 1:
                return "Slightly Monotone"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Expressive"
            elif value == 4:
                return "Very Expressive"
            else:
                return "Unknown"

        elif dimension_name == "High-Pitched vs. Low-Pitched":
            if value == 0:
                return "Very High-Pitched"
            elif value == 1:
                return "Slightly High-Pitched"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Slightly Low-Pitched"
            elif value == 4:
                return "Very Low-Pitched"
            else:
                return "Unknown"

        elif dimension_name == "Soft vs. Harsh":
            if value == -2:
                return "Very Harsh"
            elif value == -1:
                return "Slightly Harsh"
            elif value == 0:
                return "Neutral"
            elif value == 1:
                return "Slightly Soft"
            elif value == 2:
                return "Very Soft"
            else:
                return "Unknown"

        elif dimension_name == "Authenticity":
            if value == 0:
                return "Very Artificial"
            elif value == 1:
                return "Artificial"
            elif value == 2:
                return "Neutral"
            elif value == 3:
                return "Genuine"
            elif value == 4:
                return "Very Genuine"
            else:
                return "Unknown"

        elif dimension_name == "Recording Quality":
            if value == 0:
                return "Very Low Quality"
            elif value == 1:
                return "Bad Quality"
            elif value == 2:
                return "Decent Quality"
            elif value == 3:
                return "High Quality"
            elif value == 4:
                return "Very High Quality"
            else:
                return "Unknown"

        elif dimension_name == "Background Noise":
            if value == 0:
                return "No Noise"
            elif value == 1:
                return "Slight Noise"
            elif value == 2:
                return "Moderate Noise"
            elif value == 3:
                return "Intense Noise"
            else:
                return "Unknown"

        else:
            return "Unknown Dimension"


    def create_label(self, row, with_emotion=False, with_caption=True, with_detailed_caption=False, with_transcription=False):
        json_data = row
        tags = []

        if with_emotion:
            for key, value in json_data.items():
                if key in ["Affection", "Amusement", "Anger", "Astonishment/Surprise", "Awe", "Bitterness", "Concentration", "Confusion", "Contemplation", "Contempt", "Contentment", "Disappointment", "Disgust", "Distress", "Doubt", "Elation", "Embarrassment", "Emotional Numbness", "Fatigue/Exhaustion", "Fear", "Helplessness", "Hope/Enthusiasm/Optimism", "Impatience and Irritability", "Infatuation", "Interest", "Intoxication/Altered States of Consciousness", "Jealousy & Envy", "Longing", "Malevolence/Malice", "Pain", "Pleasure/Ecstasy", "Pride", "Relief", "Sadness", "Serious vs. Humorous", "Sexual Lust", "Shame", "Sourness", "Teasing", "Thankfulness/Gratitude", "Triumph", "Vulnerable vs. Emotionally Detached"]:
                    # Only add tags for emotions if value > 0
                    if value > 0:
                        translated_value = self.translate_emotion(value)
                        tags.append(f"{key}:{translated_value}")
                elif key in ["Valence", "Arousal", "Submissive vs. Dominant", "Age", "Gender", "Vulnerable vs. Emotionally Detached", "Confident vs. Hesitant", "Warm vs. Cold", "Monotone vs. Expressive", "High-Pitched vs. Low-Pitched", "Soft vs. Harsh", "Authenticity", "Recording Quality", "Background Noise"]:
                    translated_value = self.translate_dimension(key, value)
                    tags.append(f"{key}:{translated_value}")

        label = ""
        if with_emotion:
            label += "" + ",".join(tags)
        if with_caption:
            if 'caption' not in json_data or json_data['caption'] is None or len(json_data['caption']) < 10:
                if 'transcription' in json_data and json_data['transcription'] is not None:
                    label += re.sub(r"\s+\[\[.+", "", json_data['transcription'])
                else:
                    label += ""
            else:
                label += "" + json_data['caption']
        if with_detailed_caption:
            if 'detailed_caption' in json_data and json_data['detailed_caption'] is not None:
                label += "" + json_data['detailed_caption']
            else:
                # print("No detailed caption found")
                label += "" + json_data['caption']
        if with_transcription:
            label += "" + json_data['transcription']
        label = label.strip()
        return label
    