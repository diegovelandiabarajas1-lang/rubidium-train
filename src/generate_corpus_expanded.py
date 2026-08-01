#!/usr/bin/env python3
"""
RUBIDIUM - Expanded Corpus Generator (50K+ pairs)
Template Expansion + Combinatorial Generation
Target: 50K+ U:/B: pairs for training
"""
import json
import random
import os
import sys
from pathlib import Path

# ============================================================
# EXPANDED TEMPLATES (50K+ pairs)
# ============================================================

# Each topic has: questions, concepts, responses, and variations
TOPICS = {
    "programacion": {
        "questions": [
            "Que es {c}?",
            "Como funciona {c}?",
            "Para que sirve {c}?",
            "Ventajas de {c}?",
            "Como aprender {c}?",
            "Errores comunes en {c}?",
            "Mejores practicas en {c}?",
            "Diferencia entre {c} y {c2}?",
            "Futuro de {c}?",
            "Ejemplos de {c}?",
            "Por que usar {c}?",
            "Cuando usar {c}?",
            "Limitaciones de {c}?",
            "Comparar {c} con {c2}",
            "Tutorial basico de {c}",
        ],
        "concepts": [
            "Python", "JavaScript", "TypeScript", "Rust", "Go", "C++", "Java",
            "Kotlin", "Swift", "PHP", "Ruby", "Scala", "Haskell", "Elixir",
            "React", "Vue", "Angular", "Svelte", "Next.js", "Nuxt",
            "Django", "Flask", "FastAPI", "Express", "Spring Boot", "Rails",
            "Node.js", "Deno", "Bun", "Docker", "Kubernetes", "Terraform",
            "AWS", "Azure", "GCP", "Firebase", "Supabase", "PlanetScale",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "DynamoDB",
            "GraphQL", "REST", "gRPC", "WebSockets", "MQTT",
            "Git", "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
            "HTML", "CSS", "Tailwind", "SASS", "Bootstrap",
            "Linux", "Bash", "PowerShell", "Networking", "SSH",
            "Algoritmos", "Estructuras de datos", "Patrones de diseño",
            "Clean Code", "SOLID", "TDD", "BDD", "Design Patterns",
        ],
        "responses": [
            "{c} es una herramienta poderosa que permite {accion}. En la practica, se usa para {caso}.",
            "Para entender {c}, debes saber que {explicacion}. Ejemplo: {ejemplo}.",
            "{c} destaca por {ventaja}. A diferencia de {c2}, ofrece {diferencia}.",
            "Los expertos recomiendan {consejo} al trabajar con {c}. Evita {error}.",
            "El futuro de {c} apunta hacia {tendencia}. {ejemplo}.",
            "{c} se usa ampliamente en {caso}. Su principal ventaja es {ventaja}.",
            "Aprender {c} toma {tiempo}. Lo importante es {importante}.",
            "La diferencia principal entre {c} y {c2} es {diferencia}.",
            "En {caso}, {c} es la mejor opcion porque {razon}.",
            "Un error comun con {c} es {error}. Mejor {mejor}.",
        ],
        "actions": ["crear aplicaciones", "automatizar procesos", "analizar datos",
                    "desarrollar software", "construir APIs", "gestionar infraestructura"],
        "cases": ["desarrollo web", "ciencia de datos", "backend", "frontend",
                  " DevOps", "mobile", "cloud", "startup", "empresa grande"],
        "explanations": ["es un lenguaje de programacion", "es un framework moderno",
                         "es una arquitectura de software", "es un patron de diseño"],
        "examples": ["Netflix lo usa para streaming", "Google lo usa en busqueda",
                     "Spotify lo usa para recomendaciones", "Amazon lo usa en e-commerce",
                     "Instagram lo usa para posts", "WhatsApp lo usa para mensajeria"],
        "advantages": ["velocidad", "escalabilidad", "simplicidad", "rendimiento",
                       "mantenibilidad", "flexibilidad", "seguridad"],
        "others": ["alternativas tradicionales", "versiones anteriores", "competidores"],
        "differences": ["mejor rendimiento", "mas simple", "mas seguro", "mas rapido"],
        "advices": ["usar tipos estaticos", "escribir tests", "documentar codigo",
                    "hacer code review", "seguir convenciones", "practicar a diario"],
        "errors": ["no validar entrada", "hardcodear valores", "ignorar errores",
                   "sobreingenieria", "no testear", "copiar sin entender"],
        "trends": ["IA generativa", "edge computing", "WebAssembly", "Rust en backend",
                   "serverless", "microservicios", "event-driven"],
        "times": ["unas semanas", "1-3 meses", "6 meses para dominar", "1 ano para experto"],
        "importants": ["practicar a diario", "construir proyectos", "leer documentacion",
                       "aprender de errores", "ser constante"],
        "reasons": ["su ecosistema es enorme", "tiene gran comunidad", "es muy versatil"],
    },

    "ciencia": {
        "questions": [
            "Que es {c}?",
            "Por que es importante {c}?",
            "Como funciona {c}?",
            "Aplicaciones de {c}?",
            "Diferencia entre {c} y {c2}?",
            "Historia de {c}?",
            "Teoria de {c}?",
            "Investigacion reciente en {c}?",
            "Impacto de {c} en la sociedad?",
            "Futuro de {c}?",
        ],
        "concepts": [
            "ADN", "evolucion", "agujeros negros", "materia oscura", "fotosintesis",
            "mecanica cuantica", "relatividad", "celdas madre", "vacunas", "cambio climatico",
            "neuronas", "tabla periodica", "enlaces quimicos", "termodinamica",
            "astronomia", "cosmologia", "genetica", "epigenetica", "microbiologia",
            "nanotecnologia", "biotecnologia", "inteligencia artificial",
            "energia solar", "fusion nuclear", "superconductores",
            "ondas gravitacionales", "particulas subatomicas", "antimateria",
            "criogenia", "quimica organica", "bioquimica", "neurociencia",
            "psicologia", "sociologia", "economia", "filosofia",
        ],
        "responses": [
            "{c} es un fenomeno fundamental en {campo}. {explicacion}.",
            "La importancia de {c} radica en {importancia}. Sin el, {consecuencia}.",
            "Se descubrio cuando {descubrimiento}. Esto cambio {impacto}.",
            "Aplicaciones incluyen {aplicaciones}. En el futuro, {futuro}.",
            "{c} estudia {objeto}. Su relevancia es {ventaja}.",
            "Los cientificos descubrieron que {c} permite {accion}.",
            "En {campo}, {c} es fundamental porque {razon}.",
            "La teoria de {c} explica {explicacion_simple}.",
            "Investigacion reciente muestra que {c} tiene implicaciones en {aplicacion}.",
            "El impacto de {c} en la sociedad es {impacto_social}.",
        ],
        "fields": ["biologia", "fisica", "quimica", "astronomia", "neurociencia",
                   "medicina", "ecologia", "matematicas", "informatica"],
        "explanations": ["contiene informacion genetica", "explica la diversidad de vida",
                         "son regiones de gravedad extrema", "es la base de la materia"],
        "importances": ["entender la vida", "explicar el universo", "desarrollar medicinas",
                        "combatir enfermedades", "proteger el planeta"],
        "consequences": ["no habria herencia", "no entenderiamos el cosmos",
                         "no tendriamos tecnologia moderna", "no existiria la vida"],
        "discoveries": ["Watson y Crick identificaron su estructura",
                        "Einstein publico su teoria", "Cientificos observaron radiacion"],
        "impacts": ["la biologia moderna", "la fisica teorica", "la medicina"],
        "objects": ["fenomenos naturales", "procesos biologicos", "estructuras materiales"],
        "simple_explanations": ["como funciona el universo", "por que las cosas existen",
                                "el origen de la vida"],
        "applications": ["medicina personalizada", "energia limpia", "exploracion espacial",
                         "tecnologia verde", "computacion cuantica"],
        "futures": ["terapias geneticas", "computacion cuantica", "colonizacion espacial"],
        "social_impacts": ["transforma industrias", "cambia formas de vida",
                          "genera nuevas profesiones"],
        "reasons": ["su impacto es medible", "tiene aplicaciones directas",
                    "abre nuevas posibilidades"],
    },

    "vida_cotidiana": {
        "questions": [
            "Como {c}?",
            "Que necesito para {c}?",
            "Consejos para {c}?",
            "Errores al {c}?",
            "Cuanto tiempo toma {c}?",
            "Beneficios de {c}?",
            "Riesgos de {c}?",
            "Mejores practicas para {c}?",
            "Herramientas para {c}?",
            "Alternativas para {c}?",
        ],
        "concepts": [
            "cocinar sano", "hacer ejercicio", "ahorrar dinero", "dormir mejor",
            "reducir estres", "aprender idioma", "organizar casa", "leer mas",
            "meditar", "planificar semana", "beber agua", "caminar diario",
            "crear habitos", "gestionar tiempo", "mejorar productividad",
            "comunicacion efectiva", "liderazgo", "trabajo en equipo",
            "creatividad", "pensamiento critico", "inteligencia emocional",
            "finanzas personales", "inversiones", "emprendimiento",
            "relaciones interpersonales", "autoestima", "bienestar mental",
            "nutricion", "sueño", "postura", "respiracion",
        ],
        "responses": [
            "Para {c}, lo clave es {clave}. Empieza por {paso1}, luego {paso2}.",
            "Necesitas {necesario}. No requiere {no_necesario}. Lo importante es {importante}.",
            "Toma {tiempo} si {condicion}. La consistencia {resultado}.",
            "Mi consejo: {consejo}. Evita {evitar}. La paciencia {resultado_final}.",
            "Error comun: {error}. Mejor: {mejor}.",
            "{c} es beneficioso porque {beneficio}. Empieza con {paso_simple}.",
            "La mejor forma de {c} es {metodo}. Requiere {requisito}.",
            "Muchos cometen el error de {error}. En su lugar, {mejor}.",
            "El secreto de {c} es {secreto}. Con practica, {resultado}.",
            "Para empezar con {c}, necesitas {necesario}. El primer paso es {paso_simple}.",
        ],
        "keys": ["constancia", "planificacion", "simplicidad", "disfrute", "progreso gradual"],
        "steps": ["definir objetivo", "crear rutina", "empezar pequeno", "medir progreso", "ajustar"],
        "necessaries": ["motivacion", "tiempo", "espacio", "herramientas basicas", "conocimiento basico"],
        "no_necessaries": ["equipo caro", "suscripciones", "perfeccion", "mucho tiempo libre"],
        "importants": ["habito diario", "disfrutar proceso", "no rendirse", "celebrar logros"],
        "times": ["15-30 min", "unas semanas", "21 dias para habito", "3 meses para resultados"],
        "conditions": ["eres constante", "te organizas", "disfrutas el proceso"],
        "results": ["mejora gradual", "habito solido", "cambio duradero", "bienestar"],
        "advices": ["empezar hoy", "no buscar perfeccion", "disfrutar camino", "ser amable contigo"],
        "evitars": ["compararse", "exigirse demasiado", "rendirse pronto", "buscar atajos"],
        "results_finals": ["vale la pena", "transforma vida", "crea libertad"],
        "errors": ["querer todo ya", "no planificar", "compararse", "rendirse", "exigirse demasiado"],
        "betters": ["empezar pequeno", "ser constante", "disfrutar", "aprender de errores"],
        "benefits": ["mejora salud", "aumenta energia", "reduce estres", "mejora animo"],
        "methods": ["rutina diaria", "enfoque gradual", "metodos comprobados"],
        "requirements": ["poco tiempo", "voluntad", "constancia"],
        "secrets": ["empezar pequeno", "ser constante", "disfrutar el proceso"],
        "simple_steps": ["un paso a la vez", "hoy mismo", "con algo pequeno"],
    },

    "tecnologia": {
        "questions": [
            "Que es {c}?",
            "Como funciona {c}?",
            "Para que sirve {c}?",
            "Ventajas de {c}?",
            "Como aprender {c}?",
            "Errores comunes en {c}?",
            "Mejores practicas en {c}?",
            "Diferencia entre {c} y {c2}?",
            "Futuro de {c}?",
            "Ejemplos de {c}?",
            "Seguridad en {c}?",
            "Rendimiento de {c}?",
            "Escalabilidad de {c}?",
            "Costos de {c}?",
            "Alternativas a {c}?",
        ],
        "concepts": [
            "inteligencia artificial", "machine learning", "deep learning",
            "blockchain", "cloud computing", "edge computing", "IoT",
            "realidad virtual", "realidad aumentada", "5G", "6G",
            "ciberseguridad", "criptografia", "firewall", "VPN",
            "containers", "microservicios", "serverless", "API gateway",
            "data lake", "data warehouse", "big data", "analytics",
            "DevOps", "SRE", "platform engineering", "MLOps",
            "WebAssembly", "PWA", "SPAs", "SSR",
            "computacion cuantica", "neuromorfica", "optica",
            "energia renovable", "baterias", "hidrogeno verde",
            "vehiculos autonomos", "drones", "robotica",
            "biometria", "reconocimiento facial", "procesamiento de lenguaje natural",
        ],
        "responses": [
            "{c} es una tecnologia que permite {accion}. Se usa en {caso}.",
            "Para entender {c}, debes saber que {explicacion}. Ejemplo: {ejemplo}.",
            "{c} destaca por {ventaja}. A diferencia de {c2}, ofrece {diferencia}.",
            "Los expertos recomiendan {consejo} con {c}. Evita {error}.",
            "El futuro de {c} apunta hacia {tendencia}. {ejemplo}.",
            "La seguridad en {c} es {seguridad}. Por eso {razon}.",
            "El rendimiento de {c} es {rendimiento}. Comparado con {c2}, {comparacion}.",
            "Para escalar {c}, necesitas {escalabilidad}. Ejemplo: {ejemplo}.",
            "El costo de {c} es {costo}. Pero vale la pena por {ventaja}.",
            "Alternativas a {c} incluyen {alternativa}. Cada una tiene {caracteristica}.",
        ],
        "actions": ["automatizar procesos", "analizar datos", "crear experiencias",
                    "conectar dispositivos", "proteger informacion", "optimizar sistemas"],
        "cases": ["empresas", "gobierno", "salud", "educacion", "finanzas", "retail"],
        "explanations": ["es una tecnologia emergente", "es una arquitectura moderna",
                         "es un paradigma nuevo"],
        "examples": ["Amazon lo usa en logistica", "Tesla lo usa en autos",
                     "Netflix lo usa para streaming", "Google lo usa en busqueda"],
        "advantages": ["eficiencia", "escalabilidad", "seguridad", "velocidad", "automatizacion"],
        "others": ["tecnologias anteriores", "enfoques tradicionales"],
        "differences": ["mayor eficiencia", "menor complejidad", "mejor escalabilidad"],
        "advices": ["empezar con lo basico", "practicar a diario", "leer documentacion"],
        "errors": ["ignorar seguridad", "sobrecomplicar", "no medir resultados"],
        "trends": ["IA generativa", "edge computing", "quantum", "sustentabilidad"],
        "securities": ["critica", "fundamental", "prioritaria"],
        "performances": ["optimizada", "escalable", "eficiente"],
        "scalabilities": ["horizontal", "vertical", "elastica"],
        "costs": ["variable", "predecible", "escalable"],
        "alternatives": ["soluciones open source", "enfoques hibridos", "plataformas gestionadas"],
        "characteristics": ["ventajas unicas", "casos de uso especificos", "limitaciones"],
        "comparisons": ["es mas rapido", "es mas simple", "es mas seguro"],
        "reasons": ["protege datos criticos", "cumple regulaciones", "evita incidentes"],
    },

    "matematicas": {
        "questions": [
            "Que es {c}?",
            "Para que sirve {c}?",
            "Como se calcula {c}?",
            "Aplicaciones de {c}?",
            "Ejemplos de {c}?",
            "Diferencia entre {c} y {c2}?",
            "Importancia de {c}?",
            "Historia de {c}?",
            "Teoremas relacionados con {c}?",
            "Problemas con {c}?",
        ],
        "concepts": [
            "calculo diferencial", "calculo integral", "algebra lineal",
            "estadistica", "probabilidad", "geometria", "trigonometria",
            "numeros primos", "teoria de numeros", "combinatoria",
            "logica matematica", "conjuntos", "funciones", "limites",
            "derivadas", "integrales", "series", "sucesiones",
            "matrices", "determinantes", "vectores", "espacios vectoriales",
            "ecuaciones diferenciales", "analisis real", "analisis complejo",
            "topologia", "teoria de grafos", "optimizacion",
            " criptografia", "codificacion", "algoritmos",
        ],
        "responses": [
            "{c} es una rama de las matematicas que estudia {objeto}. Se aplica en {aplicacion}.",
            "La importancia de {c} radica en {importancia}. Ejemplo: {ejemplo}.",
            "Para calcular {c}, se usa {metodo}. Ejemplo: {ejemplo}.",
            "Aplicaciones incluyen {aplicaciones}. Ejemplo: {ejemplo}.",
            "El teorema mas importante de {c} es {teorema}. Explica {explicacion}.",
            "{c} se relaciona con {c2} porque {razon}.",
            "Los matematicos desarrollaron {c} para resolver {problema}.",
            "En la practica, {c} se usa para {aplicacion_practica}.",
            "Un concepto clave de {c} es {concepto_clave}. Permite {accion}.",
            "La historia de {c} comienza con {historia}. Evoluciono hacia {evolucion}.",
        ],
        "objects": ["estructuras abstractas", "relaciones cuantitativas", "patrones"],
        "applications": ["fisica", "ingenieria", "economia", "ciencias de la computacion"],
        "importances": ["resolver problemas complejos", "modelar fenomenos", "predecir resultados"],
        "examples": ["se usa en criptografia", "se aplica en IA", "se usa en finanzas"],
        "methods": ["formulas established", "algoritmos numericos", "aproximaciones"],
        "theorems": ["Pitagsoras", "Taylor", "Bayes", "Fibonacci"],
        "problems": ["calculo de areas", "optimizacion", "prediccion"],
        "practical_applications": ["diseno de algoritmos", "analisis de datos", "simulaciones"],
        "key_concepts": ["convergencia", "continuidad", "linealidad"],
        "histories": ["trabajos de Euclides", "descubrimientos de Newton", "aportes de Euler"],
        "evolutions": ["teoria moderna", "aplicaciones computacionales", "nuevas fronteras"],
        "reasons": ["comparten principios fundamentales", "se complementan mutuamente"],
    },

    "historia": {
        "questions": [
            "Que fue {c}?",
            "Cuando ocurrio {c}?",
            "Por que es importante {c}?",
            "Quienes participaron en {c}?",
            "Consecuencias de {c}?",
            "Lecciones de {c}?",
            "Contexto de {c}?",
            "Impacto de {c}?",
            "Relacion entre {c} y {c2}?",
            "Legado de {c}?",
        ],
        "concepts": [
            "Revolucion Francesa", "Revolucion Industrial", "Guerra Mundial",
            "caida del Muro de Berlin", "descubrimiento de America",
            "Renacimiento", "Ilustracion", "Reforma Protestante",
            "Revolucion Rusa", "independencia de Mexico",
            "guerra fria", "Revolucion Digital", "movimiento sufragista",
            "derechos civiles", "declaracion de derechos humanos",
            "imperio Romano", "edad media", "imperio Otomano",
            "cultura Maya", "imperio Chino", "civilizacion Griega",
            "revolucion cientifica", "era de la informacion",
        ],
        "responses": [
            "{c} fue un evento historico que ocurrio en {fecha}. Tuvo como consecuencia {consecuencia}.",
            "La importancia de {c} radica en {importancia}. Cambio {impacto}.",
            "En {c}, participaron {participantes}. Su motivacion era {motivacion}.",
            "Las consecuencias de {c} incluyen {consecuencias}. El legado es {legado}.",
            "El contexto de {c} era {contexto}. Esto llevo a {resultado}.",
            "{c} se relaciona con {c2} porque {razon}.",
            "Lecciones de {c}: {leccion}. Aplicado hoy: {aplicacion}.",
            "El impacto de {c} fue {impacto}. Cambio {cambio}.",
            "Los participantes de {c} buscaban {objetivo}. Lograron {logro}.",
            "El legado de {c} es {legado}. Se recuerda por {razon}.",
        ],
        "dates": ["siglo XVIII", "siglo XIX", "siglo XX", "edad antigua", "medieval"],
        "consequences": ["cambios sociales", "transformacion politica", "avances tecnologicos"],
        "importances": ["transformo sociedades", "cambio la historia", "definio el presente"],
        "participants": ["lideres politicos", "pueblo llano", "ejercitos", "intelectuales"],
        "motivations": ["libertad", "igualdad", "poder", "supervivencia", "justicia"],
        "contexts": ["tension politica", "crisis economica", "avance tecnologico"],
        "legacies": ["instituciones modernas", "derechos fundamentales", "cambios permanentes"],
        "impacts": ["transformo economias", "cambio culturas", "definio fronteras"],
        "changes": ["la forma de gobernar", "la sociedad", "la economia", "la cultura"],
        "objectives": ["independencia", "justicia social", "poder politico", "reforma"],
        "achievements": ["sus objetivos", "cambios duraderos", "reformas importantes"],
        "lessons": ["el poder del pueblo", "la importancia del dialogo", "conocer la historia"],
        "reasons": ["comparten contexto historico", "tienen relacion causal"],
    },

    "filosofia": {
        "questions": [
            "Que es {c}?",
            "Quien creo {c}?",
            "Por que es relevante {c}?",
            "Aplicaciones de {c}?",
            "Criticas a {c}?",
            "Diferencia entre {c} y {c2}?",
            "Ejemplos de {c}?",
            "Problemas de {c}?",
            "Relacion entre {c} y {c2}?",
            "Impacto de {c}?",
        ],
        "concepts": [
            "existencialismo", "utilitarismo", "estoicismo", "nihilismo",
            "idealismo", "materialismo", "pragmatismo", "fenomenologia",
            "etica", "logica", "epistemologia", "metafisica",
            "filosofia politica", "filosofia moral", "filosofia del lenguaje",
            "determinismo", "libre albedrio", "consequentialismo",
            "deontologia", "virtudes", "justicia", "verdad",
            "belleza", "bien", "mal", "sentido de la vida",
            "Socrates", "Platon", "Aristoteles", "Nietzsche", "Kant",
        ],
        "responses": [
            "{c} es una corriente filosofica que propone {proposicion}. Se aplica en {aplicacion}.",
            "La importancia de {c} radica en {importancia}. Cuestiona {cuestion}.",
            "{c} fue creado por {creador}. Su idea central es {idea}.",
            "Criticas a {c}: {critica}. Sin embargo, {defensa}.",
            "{c} se diferencia de {c2} en {diferencia}.",
            "En la practica, {c} se aplica en {aplicacion_practica}.",
            "Un problema central de {c} es {problema}. Los filosofos debaten {debate}.",
            "El impacto de {c} es {impacto}. Influjo en {influencia}.",
            "{c} cuestiona {cuestion}. Su respuesta es {respuesta}.",
            "Los seguidores de {c} creen que {creencia}. Esto implica {implicacion}.",
        ],
        "propositions": ["una vision del mundo", "un metodo de analisis", "una teoria del conocimiento"],
        "creators": ["pensadores griegos", "filosofos modernos", "escuelas antiguas"],
        "ideas": ["la razon como guia", "la experiencia como base", "la moral como fundamento"],
        "critics": ["falta de evidencia empirica", "simplificacion excesiva"],
        "defenses": ["su valor es conceptual", "ofrece marcos utiles"],
        "questions_philo": ["la naturaleza de la realidad", "el origen del conocimiento", "el sentido de la vida"],
        "answers": ["perspectivas multiples", "no hay respuestas absolutas", "la razon guia"],
        "implications": ["cambia nuestra perspectiva", "transforma decisiones"],
        "influences": ["la ciencia", "el arte", "la politica", "la cultura"],
        "debates": ["interpretaciones", "aplicaciones", "limites"],
        "problems": ["paradojas", "contradicciones", "ambiguedades"],
        "applications": ["etica aplicada", "decisiones morales", "analisis critico"],
    },

    "salud": {
        "questions": [
            "Que es {c}?",
            "Como prevenir {c}?",
            "Sintomas de {c}?",
            "Tratamiento de {c}?",
            "Causas de {c}?",
            "Diferencia entre {c} y {c2}?",
            "Consejos para {c}?",
            "Diagnostico de {c}?",
            "Prevencion de {c}?",
            "Rehabilitacion de {c}?",
        ],
        "concepts": [
            "diabetes", "hipertension", "colesterol alto", "ansiedad",
            "depresion", "insomnio", "migraña", "artritis",
            "osteoporosis", "anemia", "asma", "alergias",
            "obesidad", "sobrepeso", "desnutricion", "deshidratacion",
            "lesiones deportivas", "dolor lumbar", "cervicalgia",
            "burnout", "sedentarismo", "adicciones", "trastornos alimenticios",
            "vacunacion", "higiene", "nutricion", "ejercicio",
            "meditacion", "respiracion", "relajacion", "sueño saludable",
        ],
        "responses": [
            "{c} es una condicion que afecta {afectacion}. Sus causas incluyen {causa}.",
            "Para prevenir {c}, es importante {prevencion}. Tambien {consejo}.",
            "Los sintomas de {c} incluyen {sintomas}. Consulta si {consulta}.",
            "El tratamiento de {c} incluye {tratamiento}. Consulta a {profesional}.",
            "Las causas de {c} son {causas}. El factor principal es {factor}.",
            "Diferencia: {c} se manifiesta como {manifestacion}, mientras que {c2} {manifestacion2}.",
            "Consejos para {c}: {consejo}. Evita {evitar}.",
            "El diagnostico de {c} requiere {requisito}. El medico {accion_medico}.",
            "La prevencion de {c} se basa en {base}. Practica {practica}.",
            "La rehabilitacion de {c} toma {tiempo}. Incluye {inclusion}.",
        ],
        "affectations": ["el sistema cardiovascular", "el metabolismo", "la salud mental"],
        "causes": ["factores geneticos", "estilo de vida", "factores ambientales"],
        "preventions": ["ejercicio regular", " dieta balanceada", "descanso adecuado"],
        "symptoms": ["fatiga", "dolor", "malestar general", "cambios en el apetito"],
        "treatments": ["medicacion", "terapia fisica", "cambios en el estilo de vida"],
        "professionals": ["medico general", "especialista", "nutriologo"],
        "manifestations": ["dolor agudo", "malestar cronico", "limitaciones"],
        "advice": ["mantener habits saludables", "visitar al medico regularmente"],
        "evitars": ["sedentarismo", "mala alimentacion", "estres excesivo"],
        "bases": ["habitos saludables", "prevencion primaria", "educacion en salud"],
        "practices": ["rutina de ejercicio", "alimentacion consciente", "meditacion"],
        "inclusions": ["ejercicios terapeuticos", "seguimiento medico", "apoyo psicologico"],
    },
}

# Additional filler templates for combinatorial expansion
FILLER_PATTERNS = [
    "Ademas, {respuesta}",
    "Por otro lado, {respuesta}",
    "En resumen, {respuesta}",
    "Cabe mencionar que {respuesta}",
    "Es importante notar que {respuesta}",
    "Por ejemplo, {respuesta}",
    "En la practica, {respuesta}",
    "Los expertos coinciden en que {respuesta}",
    "La evidencia sugiere que {respuesta}",
    "Tambien es relevante mencionar que {respuesta}",
    "Un dato interesante es que {respuesta}",
    "Segun estudios, {respuesta}",
    "En mi experiencia, {respuesta}",
    "Lo que mucha gente no sabe es que {respuesta}",
    "Un aspecto clave es que {respuesta}",
]

CONNECTIVES = [
    "Ademas,", "Por otro lado,", "En resumen,", "Cabe mencionar que,",
    "Es importante notar que,", "Por ejemplo,", "En la practica,",
    "Los expertos coinciden en que,", "La evidencia sugiere que,",
    "Tambien,", "Asimismo,", "Por consiguiente,", "En conclusen,",
]


def fill_template(template, params):
    """Fill a template string with params, handling missing keys gracefully."""
    result = template
    for key, val in params.items():
        result = result.replace("{" + key + "}", str(val))
    # Remove any unfilled placeholders
    import re
    result = re.sub(r'\{[a-z_0-9]+\}', 'algo', result)
    return result


def generate_pairs_for_topic(topic_name, topic_data, num_pairs):
    """Generate pairs for a single topic with combinatorial expansion."""
    pairs = []
    questions = topic_data["questions"]
    concepts = topic_data["concepts"]
    responses = topic_data["responses"]

    # Get all param lists with defaults
    def get_params(name, default=["algo"]):
        return topic_data.get(name, default)

    for _ in range(num_pairs):
        q_template = random.choice(questions)
        c1 = random.choice(concepts)
        c2 = random.choice([c for c in concepts if c != c1] or [c1])

        params = {
            "c": c1, "c2": c2,
            "accion": random.choice(get_params("actions")),
            "caso": random.choice(get_params("cases")),
            "explicacion": random.choice(get_params("explanations")),
            "ejemplo": random.choice(get_params("examples")),
            "ventaja": random.choice(get_params("advantages")),
            "diferencia": random.choice(get_params("differences")),
            "consejo": random.choice(get_params("advices")),
            "error": random.choice(get_params("errors")),
            "tendencia": random.choice(get_params("trends")),
            "tiempo": random.choice(get_params("times")),
            "importante": random.choice(get_params("importants")),
            "razon": random.choice(get_params("reasons", ["su importancia"])),
            "beneficio": random.choice(get_params("benefits", ["mejora la calidad de vida"])),
            "objeto": random.choice(get_params("objects", ["fenomenos"])),
            "campo": random.choice(get_params("fields", ["la ciencia"])),
            "aplicacion": random.choice(get_params("applications", ["multiples usos"])),
            "importancia": random.choice(get_params("importances", ["el conocimiento"])),
            "consecuencia": random.choice(get_params("consequences", ["no avanzariamos"])),
            "descubrimiento": random.choice(get_params("discoveries", ["investigaciones"])),
            "impacto": random.choice(get_params("impacts", ["el campo"])),
            "aplicaciones": random.choice(get_params("applications", ["multiples usos"])),
            "futuro": random.choice(get_params("futures", ["avances"])),
            "fecha": random.choice(get_params("dates", ["epoca moderna"])),
            "participantes": random.choice(get_params("participants", ["personas importantes"])),
            "motivacion": random.choice(get_params("motivations", ["cambio social"])),
            "contexto": random.choice(get_params("contexts", ["momento historico"])),
            "legado": random.choice(get_params("legacies", ["cambios duraderos"])),
            "leccion": random.choice(get_params("lessons", ["aprender de la historia"])),
            "creador": random.choice(get_params("creators", ["pensadores"])),
            "proposicion": random.choice(get_params("propositions", ["una vision"])),
            "idea": random.choice(get_params("ideas", ["razon como guia"])),
            "critica": random.choice(get_params("critics", ["criticas"])),
            "defensa": random.choice(get_params("defenses", ["su valor"])),
            "cuestion": random.choice(get_params("questions_philo", ["realidad"])),
            "respuesta": random.choice(get_params("answers", ["perspectivas"])),
            "afectacion": random.choice(get_params("afectations", ["salud general"])),
            "prevencion": random.choice(get_params("preventions", ["habitos saludables"])),
            "sintomas": random.choice(get_params("symptoms", ["malestar"])),
            "tratamiento": random.choice(get_params("treatments", ["medicacion"])),
            "profesional": random.choice(get_params("professionals", ["medico"])),
            "requisito": random.choice(get_params("requirements", ["examenes"])),
            "base": random.choice(get_params("bases", ["prevencion"])),
            "practica": random.choice(get_params("practices", ["ejercicio"])),
            "metodo": random.choice(get_params("methods", ["tecnicas standard"])),
            "teorema": random.choice(get_params("theorems", ["teorema fundamental"])),
            "historia": random.choice(get_params("histories", ["antiguedad"])),
            "evolucion": random.choice(get_params("evolutions", ["epoca moderna"])),
            "consecuencias": random.choice(get_params("consequences", ["cambios"])),
            "problema": random.choice(get_params("problems", ["problemas complejos"])),
            "influencia": random.choice(get_params("influences", ["diversos campos"])),
            "creencia": random.choice(get_params("creences", ["en la razon"])),
            "implicacion": random.choice(get_params("implications", ["cambios"])),
            "manifestacion": random.choice(get_params("manifestations", ["sintomas"])),
            "manifestacion2": random.choice(get_params("manifestations2", ["otros sintomas"])),
            "evitar": random.choice(get_params("evitars", ["factores de riesgo"])),
            "inclusion": random.choice(get_params("inclusions", ["seguimiento"])),
            "factor": random.choice(get_params("factors", ["factores multiples"])),
            "accion_medico": random.choice(get_params("doctor_actions", ["evalua sintomas"])),
            "aplicacion_practica": random.choice(get_params("practical_applications", ["uso diario"])),
            "concepto_clave": random.choice(get_params("key_concepts", ["concepto fundamental"])),
            "resultado": random.choice(get_params("results", ["mejora"])),
            "condicion": random.choice(get_params("conditions", ["es constante"])),
            "clave": random.choice(get_params("keys", ["constancia"])),
            "paso1": random.choice(get_params("steps", ["empezar"])),
            "paso2": random.choice(get_params("steps", ["continuar"])),
            "necesario": random.choice(get_params("necessaries", ["poco"])),
            "no_necesario": random.choice(get_params("no_necessaries", ["mucho"])),
            "consejo": random.choice(get_params("advices", ["practicar"])),
            "resultado_final": random.choice(get_params("results_finals", ["vale la pena"])),
            "mejor": random.choice(get_params("betters", ["paso a paso"])),
            "paso_simple": random.choice(get_params("simple_steps", ["un paso"])),
            "metodo": random.choice(get_params("methods", ["rutina diaria"])),
            "requisito": random.choice(get_params("requirements", ["poco tiempo"])),
            "secreto": random.choice(get_params("secrets", ["constancia"])),
            "escalabilidad": random.choice(get_params("scalabilities", ["horizontal"])),
            "costo": random.choice(get_params("costs", ["variable"])),
            "alternativa": random.choice(get_params("alternatives", ["open source"])),
            "caracteristica": random.choice(get_params("characteristics", ["ventajas"])),
            "comparacion": random.choice(get_params("comparisons", ["es mejor"])),
            "seguridad": random.choice(get_params("securities", ["critica"])),
            "rendimiento": random.choice(get_params("performances", ["optimizado"])),
        }

        question = fill_template(q_template, params)
        response = fill_template(random.choice(responses), params)

        # Sometimes add a connective
        if random.random() < 0.3:
            response = random.choice(CONNECTIVES) + " " + response[0].lower() + response[1:]

        pairs.append({"user": question.strip(), "bot": response.strip(), "topic": topic_name})

    return pairs


def main():
    print("=" * 60)
    print("RUBIDIUM - Expanded Corpus Generator (50K+ pairs)")
    print("=" * 60)

    # Configuration: scale up per topic to hit 50K+
    PAIRS_PER_TOPIC = 8500  # 6 topics x 8500 = 51,000 pairs
    ALL_PAIRS = []

    for topic_name, topic_data in TOPICS.items():
        print(f"\nGenerating {topic_name}...")
        pairs = generate_pairs_for_topic(topic_name, topic_data, PAIRS_PER_TOPIC)
        ALL_PAIRS.extend(pairs)
        print(f"  {len(pairs)} pairs")

    # Deduplicate by user question
    seen = set()
    unique_pairs = []
    for p in ALL_PAIRS:
        key = p["user"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    print(f"\nTotal: {len(ALL_PAIRS)} -> Unique: {len(unique_pairs)}")

    # Shuffle
    random.shuffle(unique_pairs)

    # Save JSONL
    output_jsonl = "corpus_expanded.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for p in unique_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Save TXT format U:/B:
    output_txt = "corpus_expanded.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        for p in unique_pairs:
            f.write(f"U: {p['user']}\n")
            f.write(f"B: {p['bot']}\n\n")

    # Stats
    user_lens = [len(p["user"]) for p in unique_pairs]
    bot_lens = [len(p["bot"]) for p in unique_pairs]
    total_chars = sum(user_lens) + sum(bot_lens)

    print(f"\n{'=' * 60}")
    print(f"DONE: {len(unique_pairs)} unique pairs")
    print(f"Files:")
    print(f"  - {output_jsonl} ({os.path.getsize(output_jsonl)/1024/1024:.1f} MB)")
    print(f"  - {output_txt} ({os.path.getsize(output_txt)/1024/1024:.1f} MB)")
    print(f"\nStats:")
    print(f"  Topics: {len(TOPICS)}")
    print(f"  User: {min(user_lens)}-{max(user_lens)} chars (avg {sum(user_lens)//len(user_lens)})")
    print(f"  Bot: {min(bot_lens)}-{max(bot_lens)} chars (avg {sum(bot_lens)//len(bot_lens)})")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Est. tokens: ~{total_chars // 4:,}")

    # Also save as compact JSON for easy loading
    output_compact = "corpus_compact.json"
    compact = [{"u": p["user"], "b": p["bot"]} for p in unique_pairs]
    with open(output_compact, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)
    print(f"  - {output_compact} ({os.path.getsize(output_compact)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
