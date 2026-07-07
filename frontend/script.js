const API_URL = (typeof __API_URL__ !== "undefined" ? __API_URL__ : "http://127.0.0.1:5000") + "/predict";

const form = document.getElementById("cropForm");
const resultDiv = document.getElementById("result");
const recommendation = document.getElementById("recommendation");
const top3Div = document.getElementById("top3");
const errorDiv = document.getElementById("error");
const errorMessage = document.getElementById("errorMessage");
const loader = document.getElementById("loader");
const submitBtn = form.querySelector("button");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    resultDiv.classList.add("hidden");
    errorDiv.classList.add("hidden");
    submitBtn.disabled = true;
    loader.classList.remove("hidden");

    const payload = {
        N: parseFloat(document.getElementById("N").value),
        P: parseFloat(document.getElementById("P").value),
        K: parseFloat(document.getElementById("K").value),
        temperature: parseFloat(document.getElementById("temperature").value),
        humidity: parseFloat(document.getElementById("humidity").value),
        ph: parseFloat(document.getElementById("ph").value),
        rainfall: parseFloat(document.getElementById("rainfall").value)
    };

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (!res.ok) {
            showError(data.error || "Server error. Please try again.");
            return;
        }

        recommendation.textContent = data.recommended_crop;

        top3Div.innerHTML = "";
        data.top_3_crops.forEach((item) => {
            const el = document.createElement("div");
            el.className = "top3-item";
            el.innerHTML = `<strong>${item.crop}</strong> (${item.confidence}%)`;
            top3Div.appendChild(el);
        });

        resultDiv.classList.remove("hidden");
    } catch (err) {
        showError("Could not reach the API server. Check your connection or try again later.");
    } finally {
        submitBtn.disabled = false;
        loader.classList.add("hidden");
    }
});

function showError(msg) {
    errorMessage.textContent = msg;
    errorDiv.classList.remove("hidden");
}
