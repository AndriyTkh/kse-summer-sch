import { useEffect, useState } from "react";

/**
 * Fetch + parse a static JSON file. `optional` files that 404 resolve to null
 * (not an error) so the Phase-2 panels just hide when their script hasn't run.
 */
export function useJson<T>(path: string, optional = false) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(path)
      .then((r) => {
        if (!r.ok) {
          if (optional && r.status === 404) return null;
          throw new Error(`${path}: ${r.status}`);
        }
        return r.json();
      })
      .then((d) => {
        if (alive) setData(d);
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [path, optional]);

  return { data, error };
}
