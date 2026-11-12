#!/usr/bin/env node
/**
 * Son 5 Deploy Kuralı (Deploy Retention Rule):
 * Firebase Hosting `tradecebrail` sitesinde yalnızca son 5 sürümü (ve aktif canlı sürümü) tutar;
 * daha eski sürümleri otomatik olarak temizler.
 */

import { execSync } from "node:child_process";

const PROJECT_ID = process.env.GCP_PROJECT_ID || "adanahizlitaksi";
const SITE_ID = process.env.FIREBASE_HOSTING_SITE || "tradecebrail";
const KEEP_COUNT = Number(process.env.HOSTING_KEEP_VERSIONS || "5");
const API_BASE = "https://firebasehosting.googleapis.com/v1beta1";

function getAccessToken() {
  if (process.env.GOOGLE_OAUTH_ACCESS_TOKEN) {
    return process.env.GOOGLE_OAUTH_ACCESS_TOKEN.trim();
  }
  const cmd =
    process.platform === "win32"
      ? "gcloud.cmd auth print-access-token"
      : "gcloud auth print-access-token";
  return execSync(cmd, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

async function apiJson(url, token, method = "GET") {
  const res = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "x-goog-user-project": PROJECT_ID,
    },
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status} (${url}): ${txt.slice(0, 240)}`);
  }
  return res.status === 204 ? {} : await res.json();
}

async function main() {
  let token;
  try {
    token = getAccessToken();
  } catch (err) {
    console.warn(`[cleanup-hosting] Access token alınamadı, temizlik atlandı: ${err.message}`);
    return;
  }

  // 1) Aktif yayındaki sürümü öğren (asla silme)
  const releasesData = await apiJson(
    `${API_BASE}/sites/${SITE_ID}/releases?pageSize=5`,
    token
  );
  const activeVersionNames = new Set(
    (releasesData.releases || [])
      .slice(0, 1)
      .map((r) => r.version && r.version.name)
      .filter(Boolean)
  );

  // 2) Tüm sürümleri listele ve en yeniden en eskiye sırala
  const versionsData = await apiJson(
    `${API_BASE}/sites/${SITE_ID}/versions?pageSize=100`,
    token
  );
  const finalized = (versionsData.versions || [])
    .filter((v) => v.status === "FINALIZED" && v.name)
    .sort((a, b) => String(b.createTime || "").localeCompare(String(a.createTime || "")));

  const toKeep = new Set(finalized.slice(0, KEEP_COUNT).map((v) => v.name));
  for (const active of activeVersionNames) {
    toKeep.add(active);
  }

  const toDelete = finalized.filter((v) => !toKeep.has(v.name));
  if (toDelete.length === 0) {
    console.log(
      `[cleanup-hosting] ${SITE_ID}: Toplam ${finalized.length} sürüm var (<= ${KEEP_COUNT}); silinecek eski sürüm yok.`
    );
    return;
  }

  let deleted = 0;
  for (const ver of toDelete) {
    try {
      await apiJson(`${API_BASE}/${ver.name}`, token, "DELETE");
      deleted += 1;
    } catch (err) {
      console.warn(`[cleanup-hosting] Silinemedi (${ver.name}): ${err.message}`);
    }
  }
  console.log(
    `[cleanup-hosting] ${SITE_ID}: ${deleted} eski sürüm temizlendi; son ${KEEP_COUNT} sürüm korundu.`
  );
}

async function cleanupCloudRunRevisions() {
  const project = process.env.GCP_PROJECT_ID || "adanahizlitaksi";
  const region = process.env.CLOUD_RUN_REGION || "europe-west1";
  const service = process.env.CLOUD_RUN_SERVICE || "tradecebrail-bot";
  let token;
  try {
    token = getAccessToken();
  } catch {
    return;
  }
  const base = `https://run.googleapis.com/v2/projects/${project}/locations/${region}/services/${service}`;
  let svc;
  try {
    svc = await apiJson(base, token);
  } catch {
    return;
  }
  const activeRevs = new Set(
    (svc.trafficStatuses || svc.traffic || []).map((t) => t.revision).filter(Boolean)
  );
  const revData = await apiJson(`${base}/revisions?pageSize=50`, token);
  const revs = (revData.revisions || []).sort((a, b) =>
    String(b.createTime || "").localeCompare(String(a.createTime || ""))
  );
  const keepNames = new Set(revs.slice(0, KEEP_COUNT).map((r) => r.name));
  for (const r of revs) {
    const shortName = String(r.name || "").split("/").pop();
    if (activeRevs.has(shortName) || activeRevs.has(r.name)) {
      keepNames.add(r.name);
    }
  }
  const oldRevs = revs.filter((r) => r.name && !keepNames.has(r.name));
  for (const old of oldRevs) {
    try {
      await apiJson(`https://run.googleapis.com/v2/${old.name}`, token, "DELETE");
    } catch {
      // Aktif trafik tasiyan revizyon silinmez
    }
  }
  if (oldRevs.length > 0) {
    console.log(`[cleanup-cloudrun] ${service}: ${oldRevs.length} eski Cloud Run revizyonu temizlendi.`);
  }
}

main()
  .then(() => cleanupCloudRunRevisions())
  .catch((err) => {
    console.warn(`[cleanup-hosting] Uyarı: ${err.message}`);
  });
