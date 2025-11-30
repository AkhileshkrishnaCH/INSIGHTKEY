const btn = document.getElementById("extract-btn")
const input = document.getElementById("input-text")

const listKeywords = document.getElementById("keywords-list")
const listKeyphrases = document.getElementById("keyphrases-list")

function clearList(el) {
if (el) {
el.innerHTML = ""
}
}

function fillList(el, items) {
if (!el || !items) return
items.forEach(v => {
const li = document.createElement("li")
li.textContent = v
el.appendChild(li)
})
}



if (btn && input) {
btn.addEventListener("click", async () => {
const text = input.value

clearList(listKeywords)
clearList(listKeyphrases)

if (!text.trim()) {
alert("Warning: Please Enter Some Text")
return
}

const original = btn.textContent
btn.textContent = "Extracting..."
btn.disabled = true
btn.style.opacity = "0.6"

try {
const response = await fetch("/api/extract", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ text })
})
const data = await response.json()

if (data.error) {
alert(data.error)
return
}

fillList(listKeywords, data.keywords)
fillList(listKeyphrases, data.keyphrases)

} catch (e) {
alert("Network error while extracting.")
console.error(e)
} finally {
btn.textContent = original
btn.disabled = false
btn.style.opacity = "1"
}
})
}
