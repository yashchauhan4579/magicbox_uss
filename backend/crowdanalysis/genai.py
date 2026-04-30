# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key="AIzaSyDpJJJnJ1LdvVLBzRPau5j98jOQWxfBn2s"
    )

    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=base64.b64decode(
                        """""
                    ),
                ),
                types.Part.from_text(text="""### System Prompt

You are a specialized AI assistant for law and order and event security, tasked with analyzing crowd heatmap images from large-scale events. Your primary mission is to identify potentially dangerous crowd patterns in real-time to help prevent stampedes, crushes, and other public safety incidents.

You will be given one or more images from a camera feed. Analyze them and identify the single most critical or representative crowd pattern. Your entire response must be a **single JSON object** that **exactly** matches the provided schema. Do not output a list or an array.

Do not include any introductory text, explanations, or extra fields in your response—your entire output must be only the valid JSON object.

**Schema:**
```json
<INSERT THE JSON SCHEMA FROM THE SECTION BELOW HERE>
```

**Instructions for populating the JSON fields:**

*   **`stampedeRiskScore`**: A calculated composite risk score from 0 (No Risk) to 100 (Extreme Danger). This should be your primary output metric, derived by considering all other factors.
    *   **Calculation Heuristic:**
        *   The score must be heavily weighted by `peakHeatmapValue`. Risk increases exponentially, not linearly, with density.
        *   `riskProfile` modifies the score. A \"Critical Bottleneck\" at 0.9 density is more immediately dangerous than \"Widespread Density\" at 0.9 density and should have a higher score.
        *   `spreadWidthMeters` acts as a risk multiplier. A large spread with high density is extremely unstable and should significantly increase the score.
    *   **Score Bands:**
        *   **0-20:** Low Risk
        *   **21-50:** Medium / Growing Risk
        *   **51-80:** High Risk / Alert
        *   **81-100:** Critical Risk / Imminent Danger
*   **`alertLevel`**: Classify the situation's severity based on the `stampedeRiskScore`. Must be `\"Low\"`, `\"Medium\"`, `\"High\"`, or `\"Critical\"`.
*   **`alertThresholdCrossed`**: A boolean flag. Set to `true` if `alertLevel` is `\"High\"` or `\"Critical\"`.
*   **`title`**: A short, actionable title summarizing the situation.
*   **`description`**: A concise 1-2 sentence description of what the heatmap shows.
*   **`riskProfile`**: A technical assessment of the *type* of danger.
*   **`highlightRegions`**: An array of labeled, normalized bounding boxes `[x_min, y_min, x_max, y_max]` for areas of interest.
*   **`timestamps`**: The ISO8601 formatted date-time string(s).
*   **`cameraViews`**: The name(s) of the camera or viewpoint.
*   **`observedPatterns`**: Keywords describing the crowd's formation.
*   **`densityMetrics`**: The raw metrics used in your calculation.

Always validate your final output against the schema before responding.
"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""**Reviewing Image Composition**

I'm currently focused on the image's composition. My initial analysis is underway, examining the interplay of light and shadow within the crowd. The heatmap overlay's data points are providing further insights, especially regarding crowd density in specific areas.


**Interpreting Crowd Dynamics**

I've examined the image and heatmap. The \"singhadwara entrance\" scene presents a dense, largely static crowd at night. I've focused on extracting key details for JSON fields: time (2025-06-27T20:14:30Z) and camera view (\"singhadwara entrance\"). My attention now turns to identifying observed patterns, specifically the crowd's static nature and high density across a broad area.


**Examining Risk Factors**

I'm now assessing risk factors derived from the image analysis. The high density across a wide area is a key concern. I've noted a \"Widespread Density / Shockwave Risk.\" Moreover, the heatmap shows concentrated zones of \"near-crush density.\" My focus now is on quantifying this risk with a stampede risk score. I'll determine the probability of a dangerous event.


**Quantifying the Danger**

I'm presently focusing on calculating the stampede risk score. Based on the peak density reading of 0.92, wide spread (80 meters), and \"widespread density/shockwave risk,\" the calculated score is 94, placing it firmly in the \"Critical\" alert level. This necessitates an \"alertThresholdCrossed\" value of true.


**Determining Key Metrics**

I'm now focusing on calculating the `stampedeRiskScore`. A peak heatmap value of 0.92, indicating extreme density, and a width of 80 meters for the high-density area, are critical factors. The risk profile of \"Widespread Density / Shockwave Risk\" further amplifies the danger. The preliminary score is 94, warranting a \"Critical\" alert, hence setting `alertThresholdCrossed` to true.


**Synthesizing Findings**

I'm synthesizing the extracted data into a cohesive JSON object. The crucial parameters have been determined: a peak heatmap value of 0.92, an estimated spread of 80 meters, and a \"Widespread Density / Shockwave Risk.\" This leads to a critical stampede risk score of 94, triggering an alert. I've also prepared relevant bounding boxes for highlight regions, as well as a concise description and informative title.


**Constructing the JSON**

I'm now completing the JSON object with the calculated values. My analysis has led to a critical risk score and a \"Critical\" alert. The JSON contains key details: an accurate timestamp, the camera view name, observed patterns, and density metrics. Additionally, I've defined the risk profile as well as descriptive title and description.


"""),
                types.Part.from_text(text="""{
  \"alertLevel\": \"Critical\",
  \"alertThresholdCrossed\": true,
  \"cameraViews\": [
    \"singhadwara entrance\"
  ],
  \"densityMetrics\": {
    \"peakHeatmapValue\": 0.92,
    \"spreadWidthMeters\": 85.0
  },
  \"description\": \"Extremely high and widespread crowd density observed at the Singhadwara entrance. The large, dense formation poses a severe risk of crowd collapse or shockwave propagation.\",
  \"highlightRegions\": [
    {
      \"boundingBox\": [
        0.02,
        0.38,
        0.98,
        0.70
      ],
      \"regionLabel\": \"Widespread High-Density Zone\"
    },
    {
      \"boundingBox\": [
        0.18,
        0.40,
        0.35,
        0.55
      ],
      \"regionLabel\": \"Peak Density Hotspot\"
    }
  ],
  \"observedPatterns\": [
    \"High Density\",
    \"Widespread Congestion\",
    \"Static Crowd\"
  ],
  \"riskProfile\": \"Widespread Density / Shockwave Risk\",
  \"stampedeRiskScore\": 94,
  \"timestamps\": [
    \"2025-06-27T20:14:30Z\"
  ],
  \"title\": \"Critical Widespread Density at Entrance\"
}"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.05,
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            description = "A single, consolidated insight from a crowd heatmap analysis, featuring a composite stampede risk score and visual bounding boxes.",
            required = ["stampedeRiskScore", "alertLevel", "alertThresholdCrossed", "title", "description", "riskProfile", "highlightRegions", "timestamps", "cameraViews", "observedPatterns", "densityMetrics"],
            properties = {
                "stampedeRiskScore": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                    description = "A calculated overall risk score for a stampede (0-100), derived from all other metrics for immediate assessment.",
                ),
                "alertLevel": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    description = "Direct classification of the situation's severity, based on the risk score.",
                    enum = ["Low", "Medium", "High", "Critical"],
                ),
                "alertThresholdCrossed": genai.types.Schema(
                    type = genai.types.Type.BOOLEAN,
                    description = "A boolean flag that is true if the situation requires an immediate alert (i.e., level is High or Critical).",
                ),
                "title": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    description = "A short, actionable title summarizing the crowd situation.",
                ),
                "description": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    description = "A 1-2 sentence factual description of the observed pattern and its implication.",
                ),
                "riskProfile": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    description = "The specific technical risk profile based on crowd density and spread.",
                    enum = ["Low Risk", "Growing Congestion", "Critical Bottleneck / Crush Risk", "Widespread Density / Shockwave Risk"],
                ),
                "highlightRegions": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "An array of labeled bounding boxes to visually highlight areas of interest on the video feed.",
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required = ["regionLabel", "boundingBox"],
                        properties = {
                            "regionLabel": genai.types.Schema(
                                type = genai.types.Type.STRING,
                                description = "A label describing what the bounding box represents (e.g., 'Peak Density Zone').",
                            ),
                            "boundingBox": genai.types.Schema(
                                type = genai.types.Type.ARRAY,
                                description = "Normalized coordinates [x_min, y_min, x_max, y_max] with (0,0) at the top-left.",
                                items = genai.types.Schema(
                                    type = genai.types.Type.NUMBER,
                                ),
                            ),
                        },
                    ),
                ),
                "timestamps": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "An array of ISO8601 formatted date-time strings when this pattern was observed.",
                    items = genai.types.Schema(
                        type = genai.types.Type.STRING,
                        format = "date-time",
                    ),
                ),
                "cameraViews": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "An array of camera or viewpoint names that captured this pattern.",
                    items = genai.types.Schema(
                        type = genai.types.Type.STRING,
                    ),
                ),
                "observedPatterns": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    description = "An array of keywords describing the crowd's physical formation or behavior.",
                    items = genai.types.Schema(
                        type = genai.types.Type.STRING,
                    ),
                ),
                "densityMetrics": genai.types.Schema(
                    type = genai.types.Type.OBJECT,
                    description = "The raw quantitative metrics used for risk score calculation.",
                    required = ["peakHeatmapValue", "spreadWidthMeters"],
                    properties = {
                        "peakHeatmapValue": genai.types.Schema(
                            type = genai.types.Type.NUMBER,
                        ),
                        "spreadWidthMeters": genai.types.Schema(
                            type = genai.types.Type.NUMBER,
                        ),
                    },
                ),
            },
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
